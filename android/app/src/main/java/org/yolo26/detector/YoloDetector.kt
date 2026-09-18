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
    private var numAnchors = 0
    private var numClasses = 0
    private var outputLayout = 0  // 0 = [1, 84, N], 1 = [1, N, 84], 2 = flat

    init {
        val modelBytes = ctx.assets.open("yolo26n.onnx").use { it.readBytes() }
        session = env.createSession(modelBytes)
        labels = ctx.assets.open("labels.txt").bufferedReader().use { it.readLines() }
        numClasses = labels.size

        val dummy = FloatBuffer.allocate(1 * 3 * IMGSZ * IMGSZ)
        val dummyTensor = OnnxTensor.createTensor(env, dummy, longArrayOf(1, 3, IMGSZ.toLong(), IMGSZ.toLong()))
        session.run(Collections.singletonMap("images", dummyTensor)).use { res ->
            val raw = res[0].value
            Log.d("YOLO26", "ONNX output class: ${raw.javaClass.name}")
            if (raw is Array<*>) {
                Log.d("YOLO26", "Array depth 1, length=${raw.size}")
                if (raw.size > 0 && raw[0] is Array<*>) {
                    val second = raw[0] as Array<*>
                    Log.d("YOLO26", "Array depth 2, second.length=${second.size}")
                    if (second.size > 0 && second[0] is FloatArray) {
                        val third = second[0] as FloatArray
                        Log.d("YOLO26", "Array depth 3, third.length=${third.size}")
                        if (raw.size == 1) {
                            val dim1 = second.size
                            val dim2 = third.size
                            Log.d("YOLO26", "Shape detectado: [1, $dim1, $dim2]")
                            if (dim1 == 84 && dim2 > 100) {
                                outputLayout = 0
                                numAnchors = dim2
                            } else if (dim2 == 84 && dim1 > 100) {
                                outputLayout = 1
                                numAnchors = dim1
                            } else {
                                outputLayout = 0
                                numAnchors = max(dim1, dim2)
                            }
                        }
                    }
                } else if (raw.size == 1 && raw[0] is FloatArray) {
                    val flat = raw[0] as FloatArray
                    Log.d("YOLO26", "Flat array length=${flat.size}")
                    if (flat.size % 84 == 0) {
                        outputLayout = 2
                        numAnchors = flat.size / 84
                    }
                }
            } else if (raw is FloatArray) {
                Log.d("YOLO26", "Flat FloatArray, length=${raw.size}")
                if (raw.size % 84 == 0) {
                    outputLayout = 2
                    numAnchors = raw.size / 84
                }
            }
            Log.d("YOLO26", "Layout=$outputLayout, anchors=$numAnchors")
        }
        dummyTensor.close()
        Log.d("YOLO26", "Detector inicializado: layout=$outputLayout, anchors=$numAnchors, classes=$numClasses")
    }

    override fun close() {
        session.close()
        env.close()
    }

    fun detect(src: Bitmap): Pair<List<Detection>, Long> {
        val t0 = System.nanoTime()
        // letterbox 320x320 (calculado uma vez, fora do use)
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
        val outRaw: Array<FloatArray> = session.run(Collections.singletonMap("images", tensor)).use { res ->
            val raw = res[0].value
            val arr = mutableListOf<FloatArray>()
            when (raw) {
                is Array<*> -> {
                    if (raw.size > 0 && raw[0] is Array<*>) {
                        val second = raw[0] as Array<*>
                        if (second.size > 0 && second[0] is FloatArray) {
                            val dim1 = second.size
                            val dim2 = (second[0] as FloatArray).size
                            if (dim1 == 84 && dim2 > 100) {
                                // [84, N] -> transpor para [N, 84]
                                for (c in 0 until numAnchors) {
                                    val row = FloatArray(84)
                                    for (r in 0 until 84) {
                                        if (r < second.size && second[r] is FloatArray) {
                                            val rowData = second[r] as FloatArray
                                            if (c < rowData.size) row[r] = rowData[c]
                                        }
                                    }
                                    arr.add(row)
                                }
                            } else if (dim2 == 84 && dim1 > 100) {
                                // [N, 84]
                                for (r in 0 until numAnchors) {
                                    if (r < second.size && second[r] is FloatArray) {
                                        val rowData = second[r] as FloatArray
                                        val row = FloatArray(84)
                                        val len = min(rowData.size, 84)
                                        System.arraycopy(rowData, 0, row, 0, len)
                                        arr.add(row)
                                    } else {
                                        arr.add(FloatArray(84))
                                    }
                                }
                            } else {
                                addFlatArray(arr, raw)
                            }
                        }
                    } else if (raw.size == 1 && raw[0] is FloatArray) {
                        val flat = raw[0] as FloatArray
                        if (flat.size % 84 == 0) {
                            val n = flat.size / 84
                            for (i in 0 until n) {
                                val row = FloatArray(84)
                                System.arraycopy(flat, i * 84, row, 0, min(84, flat.size - i * 84))
                                arr.add(row)
                            }
                        }
                    }
                }
                is FloatArray -> {
                    val flat = raw as FloatArray
                    if (flat.size % 84 == 0) {
                        val n = flat.size / 84
                        for (i in 0 until n) {
                            val row = FloatArray(84)
                            System.arraycopy(flat, i * 84, row, 0, min(84, flat.size - i * 84))
                            arr.add(row)
                        }
                    }
                }
            }
            if (arr.isEmpty()) {
                Log.e("YOLO26", "Falha ao parsear saída ONNX")
                return@use arrayOf()
            }
            arr.toTypedArray()
        }
        tensor.close()

        val anchors = outRaw.size
        if (anchors == 0) {
            return emptyList<Detection>() to ((System.nanoTime() - t0) / 1_000_000)
        }

        val boxes = mutableListOf<FloatArray>()
        val scores = mutableListOf<Float>()
        val classes = mutableListOf<Int>()

        for (i in 0 until outRaw.size) {
            val row = outRaw[i]
            if (row.size < 4 + numClasses) continue
            var best = 0
            var bestScore = row[4]
            for (c in 1 until numClasses) {
                if (c < row.size && row[4 + c] > bestScore) {
                    bestScore = row[4 + c]
                    best = c
                }
            }
            if (bestScore < CONF) continue

            val cx = row[0]
            val cy = row[1]
            val w = row[2]
            val h = row[3]

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

    private fun addFlatArray(arr: MutableList<FloatArray>, raw: Array<*>) {
        // Tenta achar array flat dentro da estrutura
        if (raw.size == 1 && raw[0] is FloatArray) {
            val flat = raw[0] as FloatArray
            if (flat.size % 84 == 0) {
                val n = flat.size / 84
                for (i in 0 until n) {
                    val row = FloatArray(84)
                    System.arraycopy(flat, i * 84, row, 0, min(84, flat.size - i * 84))
                    arr.add(row)
                }
            }
        }
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val inter = max(0f, min(a[2], b[2]) - max(a[0], b[0])) *
                max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return if (ua <= 0f) 0f else inter / ua
    }
}