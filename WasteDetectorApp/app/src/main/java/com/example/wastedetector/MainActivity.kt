package com.example.wastedetector

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
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import com.example.wastedetector.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var detector: Detector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Request camera permissions
        if (!allPermissionsGranted()) {
            requestPermissions()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        detector = Detector(baseContext, MODEL, LABELS, this)

        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        // Bind camera use cases
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            var resolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()

            // TODO: handle rotations
            //val rotation = binding.main.display.rotation
            val rotation = Surface.ROTATION_0

            // Setup camera preview
            val preview = Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .setTargetRotation(rotation)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { image ->
                if (!::bitmapBuffer.isInitialized) {
                    bitmapBuffer = createBitmap(image.width, image.height)
                    }
                image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
                image.close()

                val matrix = Matrix().apply {
                    postRotate(image.imageInfo.rotationDegrees.toFloat())
                }

                val rotatedBitmap = Bitmap.createBitmap(
                    bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                    matrix, true
                )

                detector?.detect(rotatedBitmap)
            }

            cameraProvider.unbindAll()
            try {
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )
                preview.surfaceProvider = binding.previewView.surfaceProvider
            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onEmptyDetect() {
        binding.overlayView.clear()
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.overlayView.apply {
                setResults(boundingBoxes)
                invalidate()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close()
        cameraExecutor.shutdown()
    }

    // ----------------- Permissions ---------------------
    private fun requestPermissions() {
        ActivityCompat.requestPermissions(this, PERMISSIONS_REQUIRED, REQUEST_CODE_PERMISSIONS)
    }

    private fun allPermissionsGranted() = PERMISSIONS_REQUIRED.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
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

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 1
        private val PERMISSIONS_REQUIRED = arrayOf(
            android.Manifest.permission.CAMERA
        )

        private const val MODEL = "yolo11.tflite"
        private const val LABELS = "labels.txt"
    }

}