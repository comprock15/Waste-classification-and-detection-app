package com.simpledimploma.wastedetector

import android.graphics.Bitmap

interface ModelExecutor {
    fun process(image: Bitmap)
    fun close()
}