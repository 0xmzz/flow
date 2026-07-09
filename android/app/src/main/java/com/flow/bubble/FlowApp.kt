package com.flow.bubble

import android.app.Application
import android.util.Log
import java.io.PrintWriter
import java.io.StringWriter
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/** Reports uncaught exceptions to the Flow server (/clientlog) before the
 *  process dies, so phone crashes are visible in the PC log instead of silent. */
class FlowApp : Application() {
    override fun onCreate() {
        super.onCreate()
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            try {
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                Log.e("Flow", "uncaught", e)
                report("crash", "thread=${t.name}\n$sw")
                Thread.sleep(400)  // give the POST a moment to flush
            } catch (_: Throwable) {}
            prev?.uncaughtException(t, e)
        }
    }

    private fun report(tag: String, message: String) {
        val base = Prefs(this).serverUrl.trimEnd('/')
        thread {
            runCatching {
                val c = URL("$base/clientlog").openConnection() as HttpURLConnection
                c.requestMethod = "POST"
                c.doOutput = true
                c.connectTimeout = 4000
                c.setRequestProperty("Content-Type", "application/json")
                c.setRequestProperty("X-Device", "android-bubble")
                val body = org.json.JSONObject()
                    .put("tag", tag).put("message", message).toString()
                c.outputStream.use { it.write(body.toByteArray()) }
                c.responseCode
            }
        }.also { Thread.sleep(300) }
    }
}
