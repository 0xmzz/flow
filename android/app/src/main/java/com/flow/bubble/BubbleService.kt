package com.flow.bubble

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.widget.ImageView
import android.widget.Toast
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.roundToInt

class BubbleService : Service() {

    companion object {
        const val SAMPLE_RATE = 16000
        var running = false
    }

    private lateinit var wm: WindowManager
    private var bubble: ImageView? = null
    private lateinit var lp: WindowManager.LayoutParams
    private val main = Handler(Looper.getMainLooper())

    private var recorder: AudioRecord? = null
    @Volatile private var recording = false
    private var recThread: Thread? = null
    private val pcm = ByteArrayOutputStream()

    private val idle = Color.parseColor("#3D7EDB")
    private val rec = Color.parseColor("#DC3C3C")
    private val work = Color.parseColor("#F0B428")

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        running = true
        startInForeground()
        wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        addBubble()
    }

    override fun onDestroy() {
        running = false
        stopRecording(discard = true)
        bubble?.let { runCatching { wm.removeView(it) } }
        super.onDestroy()
    }

    private fun startInForeground() {
        val ch = NotificationChannel(
            "flow", "Flow bubble", NotificationManager.IMPORTANCE_MIN)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
            .createNotificationChannel(ch)
        val n: Notification = Notification.Builder(this, "flow")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Flow — hold the bubble to dictate")
            .build()
        if (Build.VERSION.SDK_INT >= 30) {
            startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(1, n)
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun addBubble() {
        val density = resources.displayMetrics.density
        val size = (60 * density).roundToInt()
        val v = ImageView(this)
        v.setImageResource(android.R.drawable.ic_btn_speak_now)
        v.setColorFilter(Color.WHITE)
        val pad = (14 * density).roundToInt()
        v.setPadding(pad, pad, pad, pad)
        setTint(v, idle)

        lp = WindowManager.LayoutParams(
            size, size,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        )
        lp.gravity = Gravity.TOP or Gravity.START
        val prefs = Prefs(this)
        lp.x = prefs.bubbleX
        lp.y = prefs.bubbleY

        val slop = ViewConfiguration.get(this).scaledTouchSlop * 2
        var downRawX = 0f; var downRawY = 0f
        var startX = 0; var startY = 0
        var dragging = false
        var pressT = 0L

        v.setOnTouchListener { view, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    downRawX = e.rawX; downRawY = e.rawY
                    startX = lp.x; startY = lp.y
                    dragging = false
                    pressT = System.currentTimeMillis()
                    startRecording(view as ImageView)
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = e.rawX - downRawX
                    val dy = e.rawY - downRawY
                    if (!dragging && (abs(dx) > slop || abs(dy) > slop)) {
                        dragging = true
                        stopRecording(discard = true)
                        setTint(view as ImageView, idle)
                    }
                    if (dragging) {
                        lp.x = startX + dx.roundToInt()
                        lp.y = startY + dy.roundToInt()
                        wm.updateViewLayout(view, lp)
                    }
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    if (dragging) {
                        val p = Prefs(this)
                        p.bubbleX = lp.x; p.bubbleY = lp.y
                    } else {
                        val held = System.currentTimeMillis() - pressT
                        finishRecording(view as ImageView, held)
                    }
                    true
                }
                else -> false
            }
        }
        wm.addView(v, lp)
        bubble = v
    }

    private fun setTint(v: ImageView, color: Int) {
        val bg = GradientDrawable()
        bg.shape = GradientDrawable.OVAL
        bg.setColor(color)
        v.background = bg
    }

    @SuppressLint("MissingPermission")
    private fun startRecording(v: ImageView) {
        if (recording) return
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val r = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBuf * 4)
        if (r.state != AudioRecord.STATE_INITIALIZED) {
            toast("Mic unavailable — grant microphone permission in the Flow app")
            return
        }
        pcm.reset()
        recorder = r
        recording = true
        r.startRecording()
        setTint(v, rec)
        v.animate().scaleX(1.15f).scaleY(1.15f).setDuration(100).start()
        recThread = thread(name = "flow-rec") {
            val buf = ByteArray(minBuf)
            while (recording) {
                val n = r.read(buf, 0, buf.size)
                if (n > 0) pcm.write(buf, 0, n)
            }
        }
    }

    private fun stopRecording(discard: Boolean) {
        if (!recording) return
        recording = false
        recThread?.join(500)
        recorder?.runCatching { stop(); release() }
        recorder = null
        if (discard) pcm.reset()
    }

    private fun finishRecording(v: ImageView, heldMs: Long) {
        stopRecording(discard = false)
        v.animate().scaleX(1f).scaleY(1f).setDuration(100).start()
        val audio = pcm.toByteArray()
        pcm.reset()
        if (heldMs < 300 || audio.size < SAMPLE_RATE / 2) { // <0.25s of audio
            setTint(v, idle)
            return
        }
        setTint(v, work)
        thread(name = "flow-post") {
            val text = transcribe(audio)
            main.post {
                setTint(v, idle)
                if (text.isNullOrBlank()) {
                    toast("Flow: transcription failed")
                } else if (!FlowAccessibilityService.insertText(this, text)) {
                    toast("Flow: copied — enable the accessibility service to auto-insert")
                }
            }
        }
    }

    private fun transcribe(pcmBytes: ByteArray): String? {
        val p = Prefs(this)
        return runCatching {
            val url = URL(p.serverUrl.trimEnd('/') + "/transcribe")
            val c = url.openConnection() as HttpURLConnection
            c.requestMethod = "POST"
            c.doOutput = true
            c.connectTimeout = 5000
            c.readTimeout = 30000
            c.setRequestProperty("Authorization", "Bearer " + p.token)
            c.setRequestProperty("Content-Type", "audio/wav")
            c.setRequestProperty("X-Device", "android-bubble")
            c.outputStream.use { it.write(wav(pcmBytes)) }
            val body = c.inputStream.bufferedReader().readText()
            JSONObject(body).getString("text")
        }.getOrNull()
    }

    private fun wav(pcmBytes: ByteArray): ByteArray {
        val total = 36 + pcmBytes.size
        val byteRate = SAMPLE_RATE * 2
        val h = ByteArrayOutputStream(44)
        fun le16(v: Int) { h.write(v and 0xff); h.write((v shr 8) and 0xff) }
        fun le32(v: Int) { le16(v and 0xffff); le16((v shr 16) and 0xffff) }
        h.write("RIFF".toByteArray()); le32(total)
        h.write("WAVE".toByteArray()); h.write("fmt ".toByteArray())
        le32(16); le16(1); le16(1); le32(SAMPLE_RATE); le32(byteRate)
        le16(2); le16(16)
        h.write("data".toByteArray()); le32(pcmBytes.size)
        h.write(pcmBytes)
        return h.toByteArray()
    }

    private fun toast(msg: String) =
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
}
