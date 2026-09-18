package org.yolo26.detector

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var preview: PreviewView
    private lateinit var foto: ImageView
    private lateinit var overlay: OverlayView
    private lateinit var status: TextView
    private var imageCapture: ImageCapture? = null
    private var detector: YoloDetector? = null
    private val bg = Executors.newSingleThreadExecutor()
    private var cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    private val galeria = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@registerForActivityResult
        contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream)?.let { analisar(it) }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        foto = findViewById(R.id.foto)
        overlay = findViewById(R.id.overlay)
        status = findViewById(R.id.status)

        findViewById<Button>(R.id.btnFoto).setOnClickListener { capturar() }
        findViewById<Button>(R.id.btnTrocarCamera).setOnClickListener { trocarCamera() }
        findViewById<Button>(R.id.btnSalvar).setOnClickListener { salvarFoto() }
        findViewById<Button>(R.id.btnGaleria).setOnClickListener { galeria.launch("image/*") }

        foto.setOnClickListener {
            foto.visibility = View.GONE
            overlay.visibility = View.GONE
            preview.visibility = View.VISIBLE
            status.text = "Modo câmera."
        }

        bg.execute {
            try {
                detector = YoloDetector(this)
                runOnUiThread { status.text = "Modelo pronto. Fotografe ou escolha da galeria." }
            } catch (e: Exception) {
                runOnUiThread { status.text = "Erro ao carregar modelo: ${e.message}" }
            }
        }
        if (temPermissao()) iniciarCamera()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        try {
            detector?.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun temPermissao() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(code: Int, perms: Array<String>, res: IntArray) {
        super.onRequestPermissionsResult(code, perms, res)
        if (temPermissao()) iniciarCamera()
        else Toast.makeText(this, "Sem câmera: use a galeria.", Toast.LENGTH_LONG).show()
    }

    private fun iniciarCamera() {
        val provider = ProcessCameraProvider.getInstance(this)
        provider.addListener({
            val previewUse = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also { it.setSurfaceProvider(preview.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            try {
                val cameraProvider = provider.get()
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, previewUse, imageCapture)
            } catch (e: Exception) {
                Log.e("YOLO26", "Falha na câmera", e)
                runOnUiThread { status.text = "Falha na câmera: ${e.message}" }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun trocarCamera() {
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA)
            CameraSelector.DEFAULT_FRONT_CAMERA
        else
            CameraSelector.DEFAULT_BACK_CAMERA
        val label = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) "Traseira" else "Frontal"
        runOnUiThread { status.text = "Câmera $label. Reiniciando…" }
        iniciarCamera()
    }

    private fun capturar() {
        val cap = imageCapture ?: return
        status.text = "Capturando…"
        cap.takePicture(bg, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bmp = proxyParaBitmap(image)
                image.close()
                runOnUiThread { analisar(bmp) }
            }
            override fun onError(e: ImageCaptureException) {
                runOnUiThread { status.text = "Erro: ${e.message}" }
            }
        })
    }

    private fun salvarFoto() {
        val bmp = ultimoBitmap ?: return
        bg.execute {
            try {
                val uri = salvarNaGaleria(bmp)
                runOnUiThread { status.text = "Salvo: $uri" }
            } catch (e: Exception) {
                runOnUiThread { status.text = "Erro ao salvar: ${e.message}" }
            }
        }
    }

    private var ultimoBitmap: Bitmap? = null

    private fun analisar(bmp: Bitmap) {
        val det = detector
        if (det == null) {
            status.text = "Carregando modelo… tente de novo em segundos."
            return
        }
        ultimoBitmap = bmp
        foto.setImageBitmap(bmp)
        preview.visibility = View.GONE
        foto.visibility = View.VISIBLE
        overlay.visibility = View.VISIBLE
        status.text = "Detectando…"
        bg.execute {
            try {
                val (dets, ms) = det.detect(bmp)
                val resumo = if (dets.isEmpty()) "nada detectado"
                else dets.groupingBy { it.label }.eachCount()
                    .entries.joinToString { "${it.key}: ${it.value}" }
                runOnUiThread {
                    overlay.setData(dets, bmp.width, bmp.height)
                    status.text = "$ms ms • ${dets.size} obj • $resumo\n(Toque na foto para voltar)"
                }
            } catch (e: Exception) {
                runOnUiThread { status.text = "Erro na detecção: ${e.message}" }
            }
        }
    }

    private fun salvarNaGaleria(bmp: Bitmap): String {
        val nome = "YOLO26n_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}.jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, nome)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/YOLO26n")
            }
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IllegalStateException("Falha ao criar entrada no MediaStore")
        contentResolver.openOutputStream(uri)?.use { out ->
            bmp.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }
        return uri.toString()
    }

    private fun proxyParaBitmap(image: ImageProxy): Bitmap {
        val bmp = if (image.format == ImageFormat.JPEG) {
            val buffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } else {
            yuvParaBitmap(image)
        }
        val m = Matrix()
        m.postRotate(image.imageInfo.rotationDegrees.toFloat())
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    private fun yuvParaBitmap(image: ImageProxy): Bitmap {
        val y = image.planes[0].buffer
        val u = image.planes[1].buffer
        val v = image.planes[2].buffer
        val ySize = y.remaining()
        val uvSize = image.width * image.height / 4
        val nv21 = ByteArray(ySize + uvSize * 2)
        y.get(nv21, 0, ySize)
        val uBytes = ByteArray(u.remaining())
        val vBytes = ByteArray(v.remaining())
        u.get(uBytes)
        v.get(vBytes)
        var pos = ySize
        for (i in 0 until uvSize) {
            nv21[pos++] = vBytes[i * vBytes.size / uvSize]
            nv21[pos++] = uBytes[i * uBytes.size / uvSize]
        }
        val out = ByteArrayOutputStream()
        YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            .compressToJpeg(Rect(0, 0, image.width, image.height), 95, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
}