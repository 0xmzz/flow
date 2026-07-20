"""Live-mode HUD state and Tk view.

The state model deliberately has no Tk dependency so its timing and buffering
rules can be covered by the invariant tests.
"""

from collections import deque
import time


class HudState:
    IDLE_S = 8.0

    def __init__(self, *, max_lines=6):
        self.enabled = False
        self.expanded = False
        self.heard = ""
        self.ignored = False
        self.thinking = deque(maxlen=max_lines)
        self.status = ""
        self.question = False
        self.awaiting_deadline = 0.0
        self.last_event = 0.0

    def set_enabled(self, enabled, now=None):
        now = time.monotonic() if now is None else now
        self.enabled = bool(enabled)
        self.expanded = self.enabled
        self.last_event = now
        if not self.enabled:
            self.question = False
            self.awaiting_deadline = 0.0

    def _touch(self, now):
        if not self.enabled:
            return False
        self.last_event = now
        self.expanded = True
        return True

    def utterance(self, text, *, ignored=False, command=None, now=None):
        now = time.monotonic() if now is None else now
        if not self._touch(now):
            return
        self.heard = (text or "").strip()
        self.ignored = bool(ignored)
        self.thinking.clear()
        self.status = ""
        self.question = False
        self.awaiting_deadline = 0.0
        if ignored:
            self.status = "Say ‘flow’ first"
        elif command is not None:
            self.append(f"Intent resolved: {command or 'wake word'}", now=now)
            self.append("Skill chosen: command router", now=now)

    def append(self, line, now=None):
        now = time.monotonic() if now is None else now
        if self._touch(now) and line:
            self.thinking.append(str(line).strip())

    def set_status(self, text, *, question=False, deadline=0.0, now=None):
        now = time.monotonic() if now is None else now
        if not self._touch(now):
            return
        self.status = (text or "").strip()
        self.question = bool(question)
        self.awaiting_deadline = float(deadline or 0.0)
        if self.status:
            self.thinking.append(f"Response: {self.status}")

    def set_awaiting(self, active, deadline=0.0, now=None):
        now = time.monotonic() if now is None else now
        if not self._touch(now):
            return
        self.question = bool(active)
        self.awaiting_deadline = float(deadline or 0.0) if active else 0.0

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        if self.question and self.awaiting_deadline and now > self.awaiting_deadline:
            self.question = False
            self.awaiting_deadline = 0.0
        if self.enabled and self.expanded and now - self.last_event >= self.IDLE_S:
            self.expanded = False
        return self.expanded

    def status_text(self, now=None):
        now = time.monotonic() if now is None else now
        if self.question and self.awaiting_deadline:
            remaining = max(0, int(self.awaiting_deadline - now + 0.999))
            return f"{self.status}  {remaining}s" if remaining else self.status
        return self.status


class HudView:
    """Borderless card owned and driven by Overlay's Tk thread."""

    BG = "#15171c"
    FG = "#ececf0"
    SUB = "#8a8f98"
    ACCENT = "#4f8ee8"

    def __init__(self, master, tk):
        self.tk = tk
        self.state = HudState()
        self.window = tk.Toplevel(master, bg=self.BG, takefocus=0)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        try:
            self.window.attributes("-alpha", 0.97)
        except Exception:
            pass
        self.window.geometry("420x260")
        self._build()
        self.window.update_idletasks()
        self._prevent_activation()

    def _label(self, master, **kwargs):
        return self.tk.Label(master, bg=self.BG, takefocus=0, **kwargs)

    def _build(self):
        self._heard_title = self._label(
            self.window, text="HEARD", fg=self.SUB, font=("Segoe UI", 8, "bold"),
            anchor="w")
        self._heard_title.pack(fill="x", padx=18, pady=(15, 2))
        self._heard = self._label(
            self.window, text="", fg=self.FG, font=("Segoe UI", 11),
            anchor="w", justify="left", wraplength=384)
        self._heard.pack(fill="x", padx=18)
        self._hint = self._label(
            self.window, text="", fg=self.SUB, font=("Segoe UI", 8), anchor="w")
        self._hint.pack(fill="x", padx=18, pady=(1, 7))
        self._thinking = self._label(
            self.window, text="", fg=self.SUB, font=("Segoe UI", 9),
            anchor="nw", justify="left", wraplength=384)
        self._thinking.pack(fill="both", expand=True, padx=18, pady=(2, 7))
        self._status = self._label(
            self.window, text="Live listening", fg=self.FG,
            font=("Segoe UI", 10, "bold"), anchor="w", justify="left",
            wraplength=384)
        self._status.pack(fill="x", padx=18, pady=(5, 15))

    def _prevent_activation(self):
        try:
            import ctypes
            hwnd = self.window.winfo_id()
            get_style = ctypes.windll.user32.GetWindowLongW
            set_style = ctypes.windll.user32.SetWindowLongW
            style = get_style(hwnd, -20)
            set_style(hwnd, -20, style | 0x08000000 | 0x00000080)
        except Exception:
            pass

    def handle(self, event):
        kind = event[0]
        if kind == "enabled":
            self.state.set_enabled(event[1])
        elif kind == "heard":
            self.state.utterance(event[1], command=event[2])
        elif kind == "ignored":
            self.state.utterance(event[1], ignored=True)
        elif kind == "response":
            self.state.set_status(
                event[1], question=self.state.question,
                deadline=self.state.awaiting_deadline)
        elif kind == "awaiting":
            self.state.set_awaiting(event[1], event[2])
        elif kind == "workflow":
            header = f"workflow: {event[1]}"
            if header not in self.state.thinking:
                self.state.append(header)
            self.state.append(f"step {event[2]}/{event[3]}: {event[4]}")
        elif kind == "agent":
            header = f"agent: {event[1]}"
            if header not in self.state.thinking:
                self.state.append(header)
            self.state.append(event[2])
        elif kind == "agent_result":
            self.state.set_status(event[1])

    def update(self):
        self.state.tick()
        if not self.state.enabled or not self.state.expanded:
            self.window.withdraw()
            return False
        self._heard.config(text=self.state.heard or "Listening…",
                           fg=self.SUB if self.state.ignored else self.FG)
        self._hint.config(text="Say ‘flow’ first" if self.state.ignored else "")
        self._thinking.config(text="\n".join(self.state.thinking))
        self._status.config(text=self.state.status_text() or "Live listening",
                            fg=self.ACCENT if self.state.question else self.FG)
        self._place()
        self.window.deiconify()
        return True

    def _place(self):
        self.window.update_idletasks()
        width, height = 420, 260
        left, top = 0, 0
        right = self.window.winfo_screenwidth()
        bottom = self.window.winfo_screenheight()
        try:
            import ctypes
            from ctypes import wintypes

            class MonitorInfo(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD),
                            ("rcMonitor", wintypes.RECT),
                            ("rcWork", wintypes.RECT),
                            ("dwFlags", wintypes.DWORD)]

            user32 = ctypes.windll.user32
            monitor = user32.MonitorFromWindow(
                user32.GetForegroundWindow(), 2)
            info = MonitorInfo(cbSize=ctypes.sizeof(MonitorInfo))
            if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                left, top = info.rcWork.left, info.rcWork.top
                right, bottom = info.rcWork.right, info.rcWork.bottom
        except Exception:
            pass
        width = min(width, max(240, right - left - 24))
        height = min(height, max(180, bottom - top - 24))
        x = right - width - 18
        y = top + max(12, (bottom - top - height) // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
