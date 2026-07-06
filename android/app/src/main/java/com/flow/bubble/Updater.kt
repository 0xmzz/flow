package com.flow.bubble

import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.core.content.FileProvider
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

object Updater {

    private fun myVersion(ctx: Context): Long =
        ctx.packageManager.getPackageInfo(ctx.packageName, 0).longVersionCode

    /** Check the Flow server for a newer APK; download + launch installer.
     *  quiet=true only surfaces anything when an update actually exists. */
    fun check(ctx: Context, quiet: Boolean) {
        val base = Prefs(ctx).serverUrl.trimEnd('/')
        thread(name = "flow-update") {
            try {
                val c = URL("$base/apk/version").openConnection() as HttpURLConnection
                c.connectTimeout = 5000
                val j = JSONObject(c.inputStream.bufferedReader().readText())
                val remote = j.getLong("versionCode")
                val mine = myVersion(ctx)
                if (remote <= mine) {
                    if (!quiet) toast(ctx, "Flow is up to date (v$mine)")
                    return@thread
                }
                toast(ctx, "Updating Flow v$mine -> v$remote ...")
                val apk = File(ctx.cacheDir, "flow-update.apk")
                (URL("$base/apk").openConnection() as HttpURLConnection).apply {
                    connectTimeout = 5000
                    inputStream.use { i -> apk.outputStream().use { i.copyTo(it) } }
                }
                val uri = FileProvider.getUriForFile(
                    ctx, ctx.packageName + ".fileprovider", apk)
                val install = Intent(Intent.ACTION_VIEW)
                    .setDataAndType(uri, "application/vnd.android.package-archive")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                            or Intent.FLAG_ACTIVITY_NEW_TASK)
                ctx.startActivity(install)
            } catch (e: Exception) {
                if (!quiet) toast(ctx, "Update check failed: ${e.message}")
            }
        }
    }

    private fun toast(ctx: Context, msg: String) {
        android.os.Handler(ctx.mainLooper).post {
            Toast.makeText(ctx, msg, Toast.LENGTH_LONG).show()
        }
    }
}
