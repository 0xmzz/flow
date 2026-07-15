package com.flow.bubble

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

/** The floating bubble: an idle mic circle that morphs into a live waveform
 *  while recording (auto-gained, like the PC overlay) and a pulse while the
 *  server transcribes. Levels are pushed from the audio thread. */
class WaveBubbleView(ctx: Context) : View(ctx) {

    enum class State { IDLE, REC, WORK }

    var state = State.IDLE
        set(v) {
            field = v
            postInvalidateOnAnimation()
        }

    /** "where the text is going" — shown above the waveform while recording. */
    var label: String = ""

    private val d = resources.displayMetrics.density
    private val nBars = 18
    private val levels = FloatArray(nBars)
    private var head = 0
    private var ref = 0.08f
    private var workPhase = 0f

    private val p = Paint(Paint.ANTI_ALIAS_FLAG)
    private val cIdle = Color.parseColor("#3D7EDB")
    private val cRec = Color.parseColor("#E0524D")
    private val cWork = Color.parseColor("#F0B428")
    private val cPill = Color.parseColor("#15171C")

    /** Peak level 0..1 from the audio thread; auto-gained then stored. */
    fun pushLevel(raw: Float) {
        ref = max(raw, max(ref * 0.93f, 0.03f))
        val norm = if (raw < 0.015f) 0f
                   else min(1.0, (raw / ref).toDouble().pow(0.6)).toFloat()
        levels[head] = norm
        head = (head + 1) % nBars
        postInvalidateOnAnimation()
    }

    override fun onDraw(c: Canvas) {
        val w = width.toFloat(); val h = height.toFloat()
        when (state) {
            State.IDLE -> drawMic(c, w, h)
            State.REC -> drawWave(c, w, h)
            State.WORK -> drawWork(c, w, h)
        }
    }

    private fun drawMic(c: Canvas, w: Float, h: Float) {
        p.style = Paint.Style.FILL
        p.color = cIdle
        c.drawCircle(w / 2, h / 2, min(w, h) / 2 - 2 * d, p)
        p.color = Color.WHITE
        val cx = w / 2; val cy = h / 2; val mw = 7 * d; val mh = 15 * d
        c.drawRoundRect(RectF(cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2 - 5 * d),
            mw / 2, mw / 2, p)
        p.style = Paint.Style.STROKE; p.strokeWidth = 2.4f * d
        c.drawArc(RectF(cx - mw, cy - 3 * d, cx + mw, cy + mh / 2), 0f, 180f, false, p)
        p.style = Paint.Style.FILL
        c.drawRect(cx - 0.6f * d, cy + mh / 2 - 3 * d, cx + 0.6f * d, cy + mh / 2 + 3 * d, p)
    }

    private fun drawWave(c: Canvas, w: Float, h: Float) {
        p.style = Paint.Style.FILL
        p.color = cPill
        c.drawRoundRect(RectF(0f, 0f, w, h), 16 * d, 16 * d, p)
        // "→ where it's going" caption at the top (if the accessibility service
        // could name it), waveform centred in the space below.
        val hasLabel = label.isNotEmpty()
        var topPad = 0f
        if (hasLabel) {
            p.color = Color.parseColor("#9AA0AA")
            p.textSize = 10 * d
            p.textAlign = Paint.Align.CENTER
            var t = "→ $label"
            while (p.measureText(t) > w - 20 * d && t.length > 4) t = t.dropLast(1)
            c.drawText(t, w / 2, 15 * d, p)
            p.textAlign = Paint.Align.LEFT
            topPad = 14 * d
        }
        val cy = topPad + (h - topPad) / 2
        p.color = cRec
        val pad = 16 * d
        val bw = 3 * d
        val gap = (w - 2 * pad - bw) / (nBars - 1) - bw
        val span = (h - topPad) * 0.66f
        for (i in 0 until nBars) {
            val lvl = levels[(head + i) % nBars]
            val bh = max(3 * d, lvl * span)
            val x = pad + i * (bw + gap)
            c.drawRoundRect(RectF(x, cy - bh / 2, x + bw, cy + bh / 2), bw / 2, bw / 2, p)
        }
    }

    private fun drawWork(c: Canvas, w: Float, h: Float) {
        p.style = Paint.Style.FILL
        p.color = cPill
        c.drawRoundRect(RectF(0f, 0f, w, h), h / 2, h / 2, p)
        workPhase += 0.16f
        p.color = cWork
        for (i in 0 until 3) {
            val a = (sin((workPhase + i * 0.6).toDouble()) * 0.5 + 0.5).toFloat()
            p.alpha = (70 + a * 185).toInt()
            c.drawCircle(w / 2 + (i - 1) * 11 * d, h / 2, 3.6f * d, p)
        }
        p.alpha = 255
        postInvalidateOnAnimation()
    }
}
