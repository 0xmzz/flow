package com.flow.bubble

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread
import kotlin.math.roundToInt

class MainActivity : Activity() {

    private lateinit var prefs: Prefs
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = Prefs(this)
        val d = resources.displayMetrics.density
        fun dp(v: Int) = (v * d).roundToInt()

        val root = LinearLayout(this)
        root.orientation = LinearLayout.VERTICAL
        root.setPadding(dp(20), dp(20), dp(20), dp(20))

        fun label(t: String) = TextView(this).also {
            it.text = t; it.setPadding(0, dp(12), 0, dp(4)); root.addView(it)
        }
        fun button(t: String, cb: () -> Unit) = Button(this).also {
            it.text = t; it.setOnClickListener { cb() }
            root.addView(it, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        }

        label("Flow server URL (over Tailscale)")
        val url = EditText(this); url.setText(prefs.serverUrl); root.addView(url)
        label("Token (server.token in config.json on the PC)")
        val tok = EditText(this); tok.setText(prefs.token); root.addView(tok)

        button("Save + test connection") {
            prefs.serverUrl = url.text.toString().trim()
            prefs.token = tok.text.toString().trim()
            thread {
                val ok = runCatching {
                    val c = URL(prefs.serverUrl.trimEnd('/') + "/health")
                        .openConnection() as HttpURLConnection
                    c.connectTimeout = 5000
                    val model = JSONObject(
                        c.inputStream.bufferedReader().readText()).optString("model")
                    "Connected — model $model"
                }.getOrElse { "FAILED: ${it.message}" }
                runOnUiThread { status.text = ok }
            }
        }

        label("Permissions — grant all three, in order")
        button("1. Microphone") {
            requestPermissions(arrayOf(android.Manifest.permission.RECORD_AUDIO), 1)
        }
        button("2. Draw over other apps") {
            startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")))
        }
        button("3. Accessibility (auto-insert text)") {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        }

        label("Bubble")
        button("Start bubble") {
            if (!Settings.canDrawOverlays(this)) {
                Toast.makeText(this, "Grant 'Draw over other apps' first",
                    Toast.LENGTH_LONG).show()
            } else {
                startForegroundService(Intent(this, BubbleService::class.java))
                status.text = "Bubble running — hold it and speak, drag to move."
            }
        }
        button("Stop bubble") {
            stopService(Intent(this, BubbleService::class.java))
            status.text = "Bubble stopped."
        }

        status = TextView(this)
        status.setPadding(0, dp(16), 0, 0)
        status.text = if (BubbleService.running) "Bubble running." else
            "Set the server, grant permissions, then start the bubble."
        root.addView(status)

        val scroll = ScrollView(this)
        scroll.addView(root)
        setContentView(scroll)
    }
}
