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

        /** Insert text into the focused editable field. Returns a human reason
         *  on failure (text is left on the clipboard either way), null on success. */
        fun insertText(ctx: Context, text: String): String? {
            val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("flow", text))
            val svc = instance
                ?: return "Accessibility service is OFF - enable Flow in Settings > Accessibility"
            val node = svc.findEditTarget()
                ?: return "No text field focused - tap into a field first (text copied)"
            val existing = node.text?.toString().orEmpty()
            val merged = when {
                existing.isBlank() -> text
                existing.endsWith(" ") || existing.endsWith("\n") -> existing + text
                else -> "$existing $text"
            }
            val args = Bundle()
            args.putCharSequence(
                AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, merged)
            if (node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args))
                return null
            // Some fields refuse SET_TEXT; try focus + paste of the clipboard.
            node.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
            if (node.performAction(AccessibilityNodeInfo.ACTION_PASTE))
                return null
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
