package org.yolo26.detector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/** Desenha as caixas sobre a imagem (mesma matemática fitCenter do ImageView). */
class OverlayView @JvmOverloads constructor(
    ctx: Context, attrs: AttributeSet? = null
) : View(ctx, attrs) {

    private var dets: List<Detection> = emptyList()
    private var imgW = 1
    private var imgH = 1

    private val boxPaint = Paint().apply {
        color = Color.rgb(0x22, 0xC5, 0x5E)
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 42f
    }
    private val bgPaint = Paint().apply {
        color = Color.rgb(0x02, 0x06, 0x17)
        alpha = 200
    }

    fun setData(detections: List<Detection>, w: Int, h: Int) {
        dets = detections
        imgW = w
        imgH = h
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (dets.isEmpty()) return
        val s = minOf(width / imgW.toFloat(), height / imgH.toFloat())
        val dx = (width - imgW * s) / 2
        val dy = (height - imgH * s) / 2
        for (d in dets) {
            val l = dx + d.box.left * s
            val t = dy + d.box.top * s
            val r = dx + d.box.right * s
            val b = dy + d.box.bottom * s
            canvas.drawRect(l, t, r, b, boxPaint)
            val txt = "${d.label} ${(d.score * 100).toInt()}%"
            val tw = textPaint.measureText(txt)
            canvas.drawRect(l, t - 52f, l + tw + 20f, t, bgPaint)
            canvas.drawText(txt, l + 10f, t - 12f, textPaint)
        }
    }
}
