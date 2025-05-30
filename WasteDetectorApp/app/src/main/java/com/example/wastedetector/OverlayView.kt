package com.example.wastedetector

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import org.tensorflow.lite.task.vision.detector.Detection

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<Detection>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        results = listOf()
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = ContextCompat.getColor(context!!, R.color.white)
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 60f

        textPaint.color = ContextCompat.getColor(context!!, R.color.black)
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 60f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    @SuppressLint("DefaultLocale")
    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            val left = it.boundingBox.left * width
            val top = it.boundingBox.top * height
            val right = it.boundingBox.right * width
            val bottom = it.boundingBox.bottom * height
            canvas.drawRect(left, top, right, bottom, boxPaint)

            val drawableText = "${it.categories[0].label} ${String.format("%.2f", it.categories[0].score)}"

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    fun setResults(detections: List<Detection>) {
        results = detections
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 16
    }
}