package com.simpledimploma.wastedetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.scale
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader

class Classifier(
    val context: Context,
    val classifierListener: ClassifierListener,
    val numThreads: Int = 2,
) : ModelExecutor {
    private lateinit var interpreter: Interpreter
    private val labels = extractLabelsFromFile(context, LABELS)

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numCategories = 0

    private val imageProcessor = createImageProcessor()

    private val modelLock = Any()
    private var isClosed = false

    init {
        setupClassifier()
    }

    fun setupClassifier() {
        val options = createOptions()
        val model = FileUtil.loadMappedFile(context, MODEL)

        interpreter = Interpreter(model, options)

        setupShape()
    }

    private fun setupShape() {
        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // Input shape is in format of [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape != null) {
            numCategories = outputShape[1]
        }
    }

    private fun createOptions(): Interpreter.Options {
        val compatList = CompatibilityList()
        return Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                this.setNumThreads(numThreads)
            }
        }
    }

    private fun createImageProcessor(): ImageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    fun extractLabelsFromFile(context: Context, labelsPath: String): List<String> {
        return try {
            context.assets.open(labelsPath).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    reader.lineSequence()
                        .filter { it.isNotBlank() }
                        .toList()
                }
            }
        } catch (e: IOException) {
            emptyList()
        }
    }

    override fun process(image: Bitmap) {
        if (isClosed) return

        try {
            synchronized(modelLock) {
                if (!isClosed) {
                    classify(image)
                }
            }
        } catch (e: Exception) {
            Log.e("Classifier", "Classification error", e)
        }
    }

    override fun close() {
        synchronized(modelLock) {
            isClosed = true
            try {
                interpreter.close()
            } catch (e: Exception) {
                Log.e("Classifier", "Error closing interpreter", e)
            }
        }
    }

    private fun classify(frame: Bitmap) {
        if (tensorWidth == 0 || tensorHeight == 0
            || numCategories == 0) {
            return
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = frame.scale(tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numCategories), OUTPUT_IMAGE_TYPE)
        interpreter.run(imageBuffer, output.buffer)

        val results = processOutput(output.floatArray)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        classifierListener.onClassify(results, inferenceTime)
    }

    private fun processOutput(outputArray: FloatArray): Category {
        return labels.mapIndexed { index, label ->
            Category(label, outputArray[index])
        }.sortedByDescending { it.score }[0]
    }


    interface ClassifierListener {
        fun onClassify(category: Category, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32

        private const val MODEL = "classification_model.tflite"
        private const val LABELS = "labels.txt"
    }

}