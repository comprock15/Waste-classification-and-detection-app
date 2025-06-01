package com.simpledimploma.wastedetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import com.simpledimploma.wastedetector.databinding.ActivityMainBinding
import kotlinx.coroutines.android.awaitFrame
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.vision.detector.Detection
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener, Classifier.ClassifierListener {
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var bitmapBuffer: Bitmap
    private var currentModel: ModelExecutor? = null
    private var isDetectionMode: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        initializeUi()
        checkCameraPermissions()
        initializeComponents()
    }

    private fun initializeUi() {
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.switchModelButton.setOnClickListener { switchModel() }
    }

    // ----------------- Permissions ---------------------

    private fun checkCameraPermissions() {
        if (!allPermissionsGranted()) {
            requestPermissions()
        }
    }

    private fun allPermissionsGranted() = PERMISSIONS_REQUIRED.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermissions() {
        ActivityCompat.requestPermissions(this, PERMISSIONS_REQUIRED, REQUEST_CODE_PERMISSIONS)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (!grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                Toast.makeText(this, "Permissions not granted", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    // ----------------- Setup camera ---------------------

    private fun initializeComponents() {
        cameraExecutor = Executors.newSingleThreadExecutor()
        initModel()
        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        // Bind camera use cases
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            setupCamera(cameraProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun setupCamera(cameraProvider: ProcessCameraProvider) {
        var resolutionSelector = createResolutionSelector()
        val rotation = getRotation()

        val preview = createPreview(resolutionSelector, rotation)
        val imageAnalysis = createImageAnalysis(resolutionSelector, rotation)

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis
            )
            preview.surfaceProvider = binding.previewView.surfaceProvider
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun createPreview(
        resolutionSelector: ResolutionSelector,
        rotation: Int
    ): Preview = Preview.Builder()
        .setResolutionSelector(resolutionSelector)
        .setTargetRotation(rotation)
        .build()

    // TODO: handle rotations
    private fun getRotation(): Int = Surface.ROTATION_0 // binding.main.display.rotation

    private fun createResolutionSelector(
        strategy: AspectRatioStrategy = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
    ): ResolutionSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(strategy)
        .build()

    private fun createImageAnalysis(
        resolutionSelector: ResolutionSelector,
        rotation: Int
    ): ImageAnalysis = ImageAnalysis.Builder()
        .setResolutionSelector(resolutionSelector)
        .setTargetRotation(rotation)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .build()
        .apply { setAnalyzer(cameraExecutor, createAnalyzer()) }

    private fun createAnalyzer() = ImageAnalysis.Analyzer { image ->
        if (!::bitmapBuffer.isInitialized) {
            bitmapBuffer = createBitmap(image.width, image.height)
        }

        processCameraImage(image).let { processedImage ->
            currentModel?.process(processedImage)
        }
    }

    private fun processCameraImage(image: ImageProxy): Bitmap {
        return image.use { img ->
            bitmapBuffer.copyPixelsFromBuffer(img.planes[0].buffer)
            Bitmap.createBitmap(
                bitmapBuffer, 0, 0,
                bitmapBuffer.width,
                bitmapBuffer.height,
                Matrix().apply {
                    postRotate(image.imageInfo.rotationDegrees.toFloat())
                },
                true
            )
        }
    }

    override fun onDetect(detections: List<Detection>, inferenceTime: Long) {
        runOnUiThread {
            binding.overlayView.apply {
                if (isDetectionMode)
                    setResults(detections)
                else
                    setResults(emptyList())
                invalidate()
            }
        }
    }

    override fun onClassify(category: Category, inferenceTime: Long) {
        runOnUiThread {
            if (!isDetectionMode)
                binding.predictionTextView.text = "${category.label} ${String.format("%.2f", category.score)}"
            else
                updatePredictionText()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        currentModel?.close()
    }

    private fun switchModel() {
        //return
        isDetectionMode = !isDetectionMode
        initModel()
        updateButtonText()
        updatePredictionText()
    }

    private fun initModel() {
        currentModel?.close()
        currentModel = if (isDetectionMode) {
            Detector(
                context = baseContext,
                detectorListener = this
            )
        } else {
            Classifier(
                context = baseContext,
                classifierListener = this
            )
        }
    }

    private fun updateButtonText() {
        binding.switchModelButton.text =
            if (isDetectionMode) getString(R.string.switch_to_classification_text)
            else getString(R.string.switch_to_detection_text)
    }

    private fun updatePredictionText() {
        binding.predictionTextView.text = ""
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 1
        private val PERMISSIONS_REQUIRED = arrayOf(
            Manifest.permission.CAMERA
        )
    }

}