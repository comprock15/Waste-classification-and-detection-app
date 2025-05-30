package com.example.wastedetector

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
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader

class Detector(
    val context: Context,
    val detectorListener: DetectorListener,
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var maxResults: Int = 3
) {
    private lateinit var interpreter: Interpreter
    private var labels = extractLabelsFromFile(context, LABELS)

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = createImageProcessor()

    init {
        setupDetector()
    }

    fun setupDetector() {
        val compatList = CompatibilityList()

        val options = Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                this.setNumThreads(numThreads)
            }
        }

        val model = FileUtil.loadMappedFile(context, MODEL)
        interpreter = Interpreter(model, options)

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
            numChannel = outputShape[1]
            numElements = outputShape[2]
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

    fun close() {
        interpreter.close()
    }

    fun detect(frame: Bitmap) {
        if (tensorWidth == 0
            || tensorHeight == 0
            || numChannel == 0
            || numElements == 0) {
            return
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = frame.scale(tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        detectorListener.onDetect(bestBoxes, inferenceTime)
    }

    private fun bestBox(array: FloatArray) : List<Detection> {

        val detections = mutableListOf<Detection>()

        for (c in 0 until numElements) {
            val categories = mutableListOf<Category>()

            for (j in 4 until numChannel) {
                categories.add(
                    Category(labels[j-4], array[c + numElements * j])
                )
            }
            categories.sortByDescending { it.score }

            if (categories[0].score > threshold) {
                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                detections.add(
                    Detection.create(
                        RectF(x1, y1, x2, y2),
                        categories
                    )
                )
            }
        }

        return applyNMS(detections).take(maxResults)
    }

    private fun applyNMS(boxes: List<Detection>) : MutableList<Detection> {
        val sortedBoxes = boxes.sortedByDescending { it.categories[0].score }.toMutableList()
        val selectedBoxes = mutableListOf<Detection>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: Detection, box2: Detection): Float {
        val left = maxOf(box1.boundingBox.left, box2.boundingBox.left)
        val bottom = maxOf(box1.boundingBox.bottom, box2.boundingBox.bottom)
        val right = minOf(box1.boundingBox.right, box2.boundingBox.right)
        val top = minOf(box1.boundingBox.top, box2.boundingBox.top)
        val intersectionArea = maxOf(0F, right - left) * maxOf(0F, bottom - top)
        val box1Area = box1.boundingBox.width() * box1.boundingBox.height()
        val box2Area = box2.boundingBox.width() * box2.boundingBox.height()
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    interface DetectorListener {
        fun onDetect(detections: List<Detection>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val IOU_THRESHOLD = 0.5F

        private const val MODEL = "yolo11.tflite"
        private const val LABELS = "labels.txt"
    }

}