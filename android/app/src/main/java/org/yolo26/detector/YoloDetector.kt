package org.yolo26.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.max
import kotlin.math.min

data class Detection(val box: RectF, val label: String, val score: Float)

class YoloDetector(ctx: Context) : AutoCloseable {
    companion object {
        const val IMGSZ = 320
        const val CONF = 0.30f
        const val IOU = 0.50f
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val labels: List<String>
    private var numClasses = 0
    private var numAnchors = 0
    private var channels = 0
    private var isTransposed = false

    init {
        val modelBytes = ctx.assets.open("yolo26n.onnx").use { it.readBytes() }
        session = env.createSession(modelBytes)
        labels = ctx.assets.open("labels.txt").bufferedReader().use { it.readLines() }
        numClasses = labels.size

        val dummy = FloatBuffer.allocate(1 * 3 * IMGSZ * IMGSZ)
        val dummyTensor = OnnxTensor.createTensor(env, dummy, longArrayOf(1, 3, IMGSZ.toLong(), IMGSZ.toLong()))
        session.run(Collections.singletonMap("images", dummyTensor)).use { res ->
            val raw = res[0].value
            Log.d("YOLO26", "=== ONNX OUTPUT SHAPE DEBUG ===")
            Log.d("YOLO26", "Class: ${raw.javaClass.name}")
            analyzeOutput(raw)
            Log.d("YOLO26", "numAnchors=$numAnchors, channels=$channels, isTransposed=$isTransposed")
        }
        dummyTensor.close()
        Log.d("YOLO26", "Detector pronto: anchors=$numAnchors, classes=$numClasses, transposed=$isTransposed")
    }

    private fun analyzeOutput(raw: Any) {
        if (raw !is Array<*>) {
            Log.e("YOLO26", "Output não é Array: ${raw.javaClass}")
            return
        }
        Log.d("YOLO26", "Depth 1: size=${raw.size}")
        if (raw.size == 0) return

        val first = raw[0]
        if (first !is Array<*>) {
            Log.d("YOLO26", "Depth 1 element is not Array: ${first?.javaClass}")
            return
        }
        val second = first as Array<*>
        Log.d("YOLO26", "Depth 2: size=${second.size}")
        if (second.size == 0) return

        val third = second[0]
        if (third !is FloatArray) {
            Log.d("YOLO26", "Depth 3 element not FloatArray: ${third?.javaClass}")
            return
        }

        val dim1 = second.size
        val dim2 = (second[0] as FloatArray).size
        Log.d("YOLO26", "Shape detectado: [1, $dim1, $dim2]")

        if (dim1 == 84 && dim2 > 100) {
            channels = 84
            numAnchors = dim2
            isTransposed = false
            Log.d("YOLO26", "Layout: [1, 84, N] -> channels=84, anchors=N")
        } else if (dim2 == 84 && dim1 > 100) {
            channels = 84
            numAnchors = dim1
            isTransposed = true
            Log.d("YOLO26", "Layout: [1, N, 84] -> channels=84, anchors=N (transposed)")
        } else {
            channels = min(dim1, dim2)
            numAnchors = max(dim1, dim2)
            isTransposed = (dim2 == 84)
            Log.w("YOLO26", "Layout inesperado, fallback: channels=$channels, anchors=$numAnchors, transposed=$isTransposed")
        }
    }

    override fun close() {
        session.close()
        env.close()
    }

    fun detect(src: Bitmap): Pair<List<Detection>, Long> {
        val t0 = System.nanoTime()

        // Letterbox calculado UMA VEZ
        val scale = min(IMGSZ / src.width.toFloat(), IMGSZ / src.height.toFloat())
        val nw = (src.width * scale).toInt()
        val nh = (src.height * scale).toInt()
        val dx = (IMGSZ - nw) / 2
        val dy = (IMGSZ - nh) / 2

        val resized = Bitmap.createScaledBitmap(src, nw, nh, true)
        val input = Bitmap.createBitmap(IMGSZ, IMGSZ, Bitmap.Config.ARGB_8888)
        input.eraseColor(0xFF727272.toInt())
        val canvas = android.graphics.Canvas(input)
        canvas.drawBitmap(resized, dx.toFloat(), dy.toFloat(), null)

        val buf = FloatBuffer.allocate(1 * 3 * IMGSZ * IMGSZ)
        val px = IntArray(IMGSZ * IMGSZ)
        input.getPixels(px, 0, IMGSZ, 0, 0, IMGSZ, IMGSZ)
        val base = IMGSZ * IMGSZ
        for (i in px.indices) {
            val c = px[i]
            buf.put(i, ((c shr 16) and 0xFF) / 255f)
            buf.put(base + i, ((c shr 8) and 0xFF) / 255f)
            buf.put(2 * base + i, (c and 0xFF) / 255f)
        }

        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, IMGSZ.toLong(), IMGSZ.toLong()))
        val output: Array<FloatArray> = session.run(Collections.singletonMap("images", tensor)).use { res ->
            val raw = res[0].value
            val rows = mutableListOf<FloatArray>()

            if (raw is Array<*>) {
                val first = raw[0]
                if (first is Array<*>) {
                    val second = first as Array<*>
                    if (second.size > 0 && second[0] is FloatArray) {
                        val dim1 = second.size
                        val dim2 = (second[0] as FloatArray).size

                        if (!isTransposed) {
                            // [84, N] -> transpose para [N, 84]
                            for (c in 0 until numAnchors) {
                                val row = FloatArray(channels)
                                for (r in 0 until channels) {
                                    if (r < second.size) {
                                        val rowData = second[r] as? FloatArray
                                        rowData?.let { if (c < it.size) row[r] = it[c] }
                                    }
                                }
                                rows.add(row)
                            }
                        } else {
                            // [N, 84] -> já está correto
                            for (r in 0 until numAnchors) {
                                if (r < second.size && second[r] is FloatArray) {
                                    val rowData = second[r] as FloatArray
                                    val row = FloatArray(channels)
                                    val len = min(rowData.size, channels)
                                    System.arraycopy(rowData, 0, row, 0, len)
                                    rows.add(row)
                                } else {
                                    rows.add(FloatArray(channels))
                                }
                            }
                        }
                    }
                }
            }
            if (rows.isEmpty()) {
                Log.e("YOLO26", "Falha ao parsear saída - rows vazio")
                return@use arrayOf()
            }
            rows.toTypedArray()
        }
        tensor.close()

        if (output.isEmpty()) {
            return emptyList<Detection>() to ((System.nanoTime() - t0) / 1_000_000)
        }

        val boxes = mutableListOf<FloatArray>()
        val scores = mutableListOf<Float>()
        val classes = mutableListOf<Int>()

        for (i in 0 until output.size) {
            val row = output[i]
            if (row.size < 4 + numClasses) continue

            var best = 0
            var bestScore = row[4]
            for (c in 1 until numClasses) {
                val idx = 4 + c
                if (idx < row.size && row[idx] > bestScore) {
                    bestScore = row[idx]
                    best = c
                }
            }
            if (bestScore < CONF) continue

            val cx = row[0]
            val cy = row[1]
            val w = row[2]
            val h = row[3]

            // cxcywh (letterbox) -> x1y1x2y2 (original) com clamp
            val x1 = (cx - w / 2 - dx) / scale
            val y1 = (cy - h / 2 - dy) / scale
            val x2 = (cx + w / 2 - dx) / scale
            val y2 = (cy + h / 2 - dy) / scale

            val x1c = max(0f, min(src.width.toFloat(), x1))
            val y1c = max(0f, min(src.height.toFloat(), y1))
            val x2c = max(0f, min(src.width.toFloat(), x2))
            val y2c = max(0f, min(src.height.toFloat(), y2))

            if (x2c > x1c && y2c > y1c) {
                boxes.add(floatArrayOf(x1c, y1c, x2c, y2c))
                scores.add(bestScore)
                classes.add(best)
            }
        }

        // NMS por classe
        val keep = mutableListOf<Int>()
        val order = scores.indices.sortedByDescending { scores[it] }.toMutableList()
        while (order.isNotEmpty()) {
            val i = order.removeAt(0)
            keep.add(i)
            val it = order.iterator()
            while (it.hasNext()) {
                val j = it.next()
                if (classes[i] == classes[j] && iou(boxes[i], boxes[j]) > IOU) it.remove()
            }
        }

        val dets = keep.map { i ->
            val b = boxes[i]
            Detection(
                RectF(b[0], b[1], b[2], b[3]),
                labels.getOrElse(classes[i]) { "obj" },
                scores[i]
            )
        }
        return dets to (System.nanoTime() - t0) / 1_000_000
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val inter = max(0f, min(a[2], b[2]) - max(a[0], b[0])) *
                max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return if (ua <= 0f) 0f else inter / ua
    }
}