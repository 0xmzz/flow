package com.flow.bubble

import android.accessibilityservice.AccessibilityService
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo

class FlowAccessibilityService : AccessibilityService() {

    companion object {
        private var instance: FlowAccessibilityService? = null

        fun isEnabled() = instance != null

        fun imeVisible(): Boolean =
            instance?.windows?.any { it.type == AccessibilityWindowInfo.TYPE_INPUT_METHOD }
                ?: false

        /** Human 'where the text is going' — app name (+ field hint) of the
         *  focused editable, for the bubble's context line. "" if unknown. */
        fun currentTarget(ctx: Context): String {
            val svc = instance ?: return ""
            val node = svc.findEditTarget()
            val pkg = node?.packageName?.toString()
                ?: svc.rootInActiveWindow?.packageName?.toString()
                ?: return ""
            val app = try {
                val pm = ctx.packageManager
                pm.getApplicationLabel(pm.getApplicationInfo(pkg, 0)).toString()
            } catch (_: Exception) {
                pkg.substringAfterLast('.')
            }
            val hint = node?.hintText?.toString()?.trim()?.takeIf { it.isNotEmpty() }
            return if (hint != null) "$app · $hint" else app
        }

        private fun fieldText(node: AccessibilityNodeInfo): String {
            node.refresh()
            return when {
                node.isShowingHintText -> ""
                node.text?.toString() == node.hintText?.toString() -> ""
                else -> node.text?.toString().orEmpty()
            }
        }

        /** Insert text into the focused editable field, VERIFYING the field
         *  actually changed — performAction can return true while doing
         *  nothing (the silent-failure mode). Returns a reason on failure. */
        fun insertText(ctx: Context, text: String): String? {
            val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("flow", text))
            val svc = instance
                ?: return "Accessibility service is OFF - enable Flow in Settings > Accessibility"
            val node = svc.findEditTarget()
                ?: return "No text field focused - tap into a field first (text copied)"
            val before = fieldText(node)
            val probe = text.trim().takeLast(12)

            fun landed(): Boolean {
                val now = fieldText(node)
                return now != before && (probe.isEmpty() || now.contains(probe))
            }

            // 1) PASTE at the cursor (hint-immune, cursor-respecting)
            node.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
            if (node.performAction(AccessibilityNodeInfo.ACTION_PASTE)) {
                Thread.sleep(120)
                if (landed()) return null
            }
            // 2) SET_TEXT merge (paste claimed success but the field didn't change,
            //    or paste was refused)
            val merged = when {
                before.isBlank() -> text
                before.endsWith(" ") || before.endsWith("\n") -> before + text
                else -> "$before $text"
            }
            val args = Bundle()
            args.putCharSequence(
                AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, merged)
            if (node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) {
                Thread.sleep(120)
                if (landed()) return null
                return "insert silently ignored by this app (pkg=${node.packageName})"
            }
            return "This app blocks text insertion (text copied - long-press to paste)"
        }
    }

    /** Focused input node, with a fallback sweep for editable+focused nodes. */
    private fun findEditTarget(): AccessibilityNodeInfo? {
        findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            ?.takeIf { it.isEditable }?.let { return it }
        for (w in windows) {
            val root = w.root ?: continue
            root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
                ?.takeIf { it.isEditable }?.let { return it }
            findEditable(root, 0)?.let { return it }
        }
        return null
    }

    private fun findEditable(node: AccessibilityNodeInfo?, depth: Int): AccessibilityNodeInfo? {
        if (node == null || depth > 12) return null
        if (node.isEditable && node.isFocused) return node
        for (i in 0 until node.childCount) {
            findEditable(node.getChild(i), depth + 1)?.let { return it }
        }
        return null
    }

    override fun onServiceConnected() {
        instance = this
        BubbleService.refreshVisibility()
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        super.onDestroy()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // Keyboard shown/hidden and focus changes both land here.
        BubbleService.refreshVisibility()
    }

    override fun onInterrupt() {}
}
