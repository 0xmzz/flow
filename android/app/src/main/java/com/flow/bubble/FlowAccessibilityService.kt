package com.flow.bubble

import android.accessibilityservice.AccessibilityService
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

class FlowAccessibilityService : AccessibilityService() {

    companion object {
        private var instance: FlowAccessibilityService? = null

        /** Insert text into the focused editable field. Returns true on success;
         *  on failure the text is left on the clipboard so a manual paste works. */
        fun insertText(ctx: Context, text: String): Boolean {
            val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("flow", text))
            val svc = instance ?: return false
            val node = svc.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) ?: return false
            if (!node.isEditable) return false
            val existing = node.text?.toString().orEmpty()
            val merged = when {
                existing.isBlank() -> text
                existing.endsWith(" ") -> existing + text
                else -> "$existing $text"
            }
            val args = Bundle()
            args.putCharSequence(
                AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, merged)
            var ok = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
            if (!ok) {
                // Fallback: paste from the clipboard we just set.
                ok = node.performAction(AccessibilityNodeInfo.ACTION_PASTE)
            }
            return ok
        }
    }

    override fun onServiceConnected() {
        instance = this
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        super.onDestroy()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}
}
