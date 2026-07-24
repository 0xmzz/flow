package com.flow.bubble

import android.content.Context

class Prefs(ctx: Context) {
    private val p = ctx.getSharedPreferences("flow", Context.MODE_PRIVATE)

    var serverUrl: String
        get() = p.getString("url", "")!!
        set(v) = p.edit().putString("url", v).apply()

    var token: String
        get() = p.getString("token", "")!!
        set(v) = p.edit().putString("token", v).apply()

    var fallbackUrl: String
        get() = p.getString("fallback_url", "")!!
        set(v) = p.edit().putString("fallback_url", v).apply()

    var bubbleX: Int
        get() = p.getInt("bx", 8)
        set(v) = p.edit().putInt("bx", v).apply()

    var bubbleY: Int
        get() = p.getInt("by", 400)
        set(v) = p.edit().putInt("by", v).apply()
}
