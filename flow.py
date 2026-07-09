"""Flow — local push-to-talk dictation for Windows.

Hold the hotkey, speak, release: transcription is typed at your cursor.
ASR runs fully local on GPU (faster-whisper). Optional LLM cleanup pass.
"""

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

# The Intel Fortran runtime bundled in numpy/ctranslate2 registers a console
# control handler that ABORTS the process on a console CLOSE/LOGOFF event —
# fatal for a background app. Disable it before those libs load.
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

# Expose pip-installed NVIDIA cuBLAS/cuDNN DLLs to ctranslate2 before import.
_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if _nvidia.is_dir():
    for _bin in _nvidia.glob("*/bin"):
        os.add_dll_directory(str(_bin))
        os.environ["PATH"] = str(_bin) + os.pathsep + os.environ["PATH"]

if sys.stdout:
    sys.stdout.reconfigure(line_buffering=True)
else:
    # pythonw.exe has no console — keep diagnostics by logging to file.
    try:
        _logf = open(Path(__file__).parent / "flow.log", "a",
                     buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = _logf
    except Exception:
        pass

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel

import llm
from store import TranscriptStore, resample_linear, save_wav
from commands import Router, speak

APP_DIR = Path(__file__).parent
CONFIG_PATH = APP_DIR / "config.json"


def _install_crash_logging():
    """Persist every uncaught exception (main + threads) to data/flow-crash.log,
    which survives restarts — flow.log is truncated on each launch."""
    import datetime
    import traceback
    crash = APP_DIR / "data" / "flow-crash.log"
    crash.parent.mkdir(parents=True, exist_ok=True)

    def record(kind, exc_type, exc, tb):
        try:
            with crash.open("a", encoding="utf-8") as f:
                f.write(f"\n===== {datetime.datetime.now().isoformat()} "
                        f"({kind}) =====\n")
                traceback.print_exception(exc_type, exc, tb, file=f)
        except Exception:
            pass
        traceback.print_exception(exc_type, exc, tb)

    sys.excepthook = lambda et, e, tb: record("main", et, e, tb)
    threading.excepthook = lambda a: record(
        f"thread:{a.thread.name}", a.exc_type, a.exc_value, a.exc_traceback)


_install_crash_logging()

DEFAULTS = {
    "hotkey": "f8",                # hold to dictate (paste transcript at cursor)
    "command_hotkey": "right shift",  # hold to issue a spoken COMMAND (acts + speaks back)
    "model": "large-v3",           # full v3 handles whispered/quiet speech better than turbo
    "device": "cuda",
    "compute_type": "float16",
    "language": None,              # None = auto-detect
    "sample_rate": 16000,
    "cleanup": {
        "enabled": False,          # LLM pass: strip filler, fix punctuation
        "provider": "anthropic",   # "anthropic" or "openai" (any OpenAI-compatible endpoint)
        "model": "claude-haiku-4-5-20251001",
        "base_url": None,          # for openai-compatible endpoints
        "api_key_env": "FLOW_API_KEY",
        "api_key": "",             # inline key (config.json is gitignored); wins over env
        "extra_body": {},          # merged into the API request (e.g. z.ai thinking off)
        # Same-shape block tried when the local model fails (Ollama down, etc.):
        # {"provider": "anthropic", "base_url": ..., "model": ..., "api_key": ...}
        "fallback": {},
        # App-aware styling: match against the foreground window title
        # (case-insensitive substring). mode "matched" = only run the LLM pass
        # for windows that match a profile (keeps raw speed elsewhere);
        # "all" = every dictation gets the neutral cleanup at minimum.
        "mode": "matched",
        "min_words": 5,            # skip the LLM pass for shorter utterances
        "profiles": [
            {"match": ["whatsapp", "telegram", "discord"],
             "style": "Casual chat message: relaxed tone, contractions fine, "
                      "minimal punctuation, no capital-letter stiffness."},
            {"match": ["claude", "code", "terminal", "powershell", "cursor"],
             "style": "Technical instruction for an AI coding agent: precise, "
                      "unambiguous, keep exact file names/paths/symbols verbatim."},
            {"match": ["gmail", "outlook", "mail"],
             "style": "Professional email prose: complete sentences, courteous, "
                      "properly punctuated."},
        ],
    },
    "paste_mode": "clipboard",     # "clipboard" (fast, restores clipboard) or "type"
    "mute_output_while_recording": True,  # silence speakers during dictation
    "min_duration_s": 0.6,         # includes the pre-roll
    "pre_roll_s": 0.4,             # mic ring buffer prepended on press
    "overlay": True,               # on-screen listening/transcribing pill
    "save_last_audio": False,      # debug: write each recording to last.wav for tuning
    # ASR tuning — the VAD gate is what drops quiet/whispered audio; keep it on
    # but permissive so whispers reach the decoder. (Verified via research pass.)
    "vad_filter": True,
    "vad_threshold": 0.3,          # was 0.5; admits whispered onsets (sweep 0.2-0.3)
    "vad_min_silence_ms": 400,     # push-to-talk clips rarely have 2s internal gaps
    "vad_speech_pad_ms": 600,      # recover faint on/offset syllables
    "condition_on_previous_text": False,  # avoid hallucination loops on short dictation
    "no_speech_threshold": None,   # let VAD gate silence, not this (None disables the drop)
    "beam_size": 3,                # final decode; streaming passes stay greedy
    "normalize_gain": False,       # peak-normalize before ASR (low value; A/B only)
    # Personal vocabulary — biases recognition toward YOUR words, so mumbled
    # domain terms and names resolve correctly (the Wispr trick).
    "vocab": [],
    "replacements": {},            # whole-word text replacements (btw -> by the way)
    "snippets": {},                # phrase -> expansion (imported from Wispr; not yet wired)
    "streaming": {
        "enabled": True,           # transcribe while the hotkey is held
        "interval_s": 0.6,         # rolling-pass cadence (a pass costs ~0.25s on the 5090)
        "unstable_tail_s": 0.6,    # words this close to the buffer edge may still change
        "pause_gap_s": 0.25,       # only cut audio at a gap this long (format integrity)
        "live_type": True,         # paste committed words WHILE you talk
                                   # (only in windows with no style profile —
                                   # styled text must wait for the full utterance)
    },
    "history": {
        "enabled": True,           # store every dictation locally
        "dir": "data",             # relative to this file
        "archive_audio": False,    # also keep paired audio for a fine-tuning corpus
    },
    "server": {
        "enabled": False,          # expose an ASR endpoint for phones over Tailscale
        "host": "0.0.0.0",         # bind all interfaces (reach via tailnet IP)
        "port": 8756,
        "token": "",               # auto-generated on first run if empty
    },
}

CLEANUP_PROMPT = (
    "You clean up raw speech-to-text dictation. Remove filler words (um, uh, like, "
    "you know), fix punctuation and capitalization, and correct obvious transcription "
    "errors from context. If the speaker is clearly dictating a list or steps, format "
    "them as list lines. Do NOT rephrase, summarize, or add anything. Preserve the "
    "speaker's wording and meaning exactly. Reply with only the cleaned text."
)


def apply_replacements(text, cfg):
    """Whole-word, case-insensitive replacements (e.g. btw -> by the way)."""
    reps = cfg.get("replacements") or {}
    for k, v in reps.items():
        text = re.sub(rf"\b{re.escape(k)}\b", lambda _m: v, text, flags=re.IGNORECASE)
    return text


def match_profile(cfg, window):
    w = (window or "").lower()
    for p in cfg["cleanup"].get("profiles", []):
        if any(m in w for m in p.get("match", [])):
            return p.get("style")
    return None


def load_config():
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if CONFIG_PATH.exists():
        user = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    else:
        CONFIG_PATH.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
    return cfg


def _persist_config(patch):
    """Merge a shallow patch into config.json on disk (one level deep)."""
    data = {}
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


class Recorder:
    """Always-open mic stream with a pre-roll ring buffer.

    Opening the stream on keypress loses ~100-300ms — the first syllable.
    Instead the stream runs permanently: while idle, audio goes into a small
    rolling buffer (RAM only, discarded continuously); on start() the last
    `pre_roll_s` seconds are prepended, so the recording begins in the past.
    """

    def __init__(self, sample_rate, pre_roll_s=0.4):
        from collections import deque
        self.sample_rate = sample_rate
        self.pre_roll = int(pre_roll_s * sample_rate)
        self._chunks = []
        self._ring = deque()
        self._ring_samples = 0
        self._recording = False
        self._stream = None
        self._open()

    def _open(self):
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32",
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            print(f"[mic] open failed: {e}")

    def _on_audio(self, indata, *_):
        chunk = indata.copy()
        if self._recording:
            self._chunks.append(chunk)
        else:
            self._ring.append(chunk)
            self._ring_samples += len(chunk)
            while (self._ring and
                   self._ring_samples - len(self._ring[0]) >= self.pre_roll):
                self._ring_samples -= len(self._ring.popleft())

    def start(self):
        if self._stream is None or not self._stream.active:
            self._open()  # recover from device loss
        self._chunks = list(self._ring)  # pre-roll: capture speech before the press
        self._recording = True

    def snapshot(self):
        """Audio captured so far (recording continues)."""
        chunks = list(self._chunks)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).flatten()

    def stop(self):
        self._recording = False
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).flatten()


class StreamWorker:
    """Transcribes rolling chunks WHILE the hotkey is held, committing text that
    is stable across two consecutive passes (LocalAgreement). At release only
    the uncommitted tail needs decoding, so release-to-paste stays ~constant no
    matter how long the dictation ran."""

    def __init__(self, flow, live=False):
        self.flow = flow
        self.committed_text = ""
        self.committed_samples = 0
        self.live = live         # paste committed words while still recording
        self.typed_len = 0       # chars of committed_text already pasted
        self.saved_clip = None
        self._prev = []          # word texts from the previous pass
        self._active = False
        self._thread = None
        if live:
            try:
                self.saved_clip = pyperclip.paste()
            except Exception:
                pass

    def start(self):
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._active = False
        if self._thread:
            self._thread.join(2.0)
        return self.committed_text, self.committed_samples

    def _loop(self):
        sr = self.flow.cfg["sample_rate"]
        scfg = self.flow.cfg["streaming"]
        interval = scfg.get("interval_s", 0.6)
        unstable = scfg.get("unstable_tail_s", 0.6)
        pause_gap = scfg.get("pause_gap_s", 0.25)
        while self._active:
            time.sleep(interval)
            if not self._active:
                return
            if self.flow.model_lock.locked():
                continue  # don't queue behind other GPU work; try next cycle
            snap = self.flow.recorder.snapshot()
            tail = snap[self.committed_samples:]
            if len(tail) < sr * 1.5:
                continue
            try:
                segs = self.flow.decode(tail, beam_size=1, word_timestamps=True,
                                        prompt=self.flow.bias_prompt(self.committed_text))
            except Exception as e:
                print(f"[stream] pass failed: {e}")
                continue
            if not self._active:   # released mid-pass: let the final pass own it
                return
            tail_dur = len(tail) / sr
            # Word-level commit: whisper often yields ONE growing segment, so
            # segment-level stability never triggers on normal sentences.
            words = [w for s in segs for w in (s.words or [])
                     if w.end < tail_dur - unstable]
            texts = [w.word.strip() for w in words]
            # Commit the longest prefix of words two consecutive passes agree on —
            # but only cut at a natural pause (>=0.35s gap), so the audio seam
            # lands on silence and phrase-level formatting ("10%", caps) survives.
            agree = 0
            for a, b in zip(texts, self._prev):
                if a == b:
                    agree += 1
                else:
                    break
            cut = 0
            for i in range(agree):
                nxt = words[i + 1].start if i + 1 < len(words) else tail_dur
                if nxt - words[i].end >= pause_gap:
                    cut = i + 1
            if cut:
                phrase = "".join(w.word for w in words[:cut]).strip()
                self.committed_text = (self.committed_text + " " + phrase).strip()
                self.committed_samples += int(words[cut - 1].end * sr)
                self._prev = texts[cut:]
                if self.live:
                    self.type_chunk()
            else:
                self._prev = texts

    def type_chunk(self):
        """Paste newly committed words at the cursor while still recording.
        Uses clipboard+ctrl+v: the user is physically holding ctrl already, so
        synthetic character typing would trigger app shortcuts — paste is safe."""
        chunk = self.committed_text[self.typed_len:]
        if not chunk:
            return
        try:
            pyperclip.copy(apply_replacements(chunk, self.flow.cfg))
            time.sleep(0.02)
            keyboard.send("ctrl+v")
            self.typed_len = len(self.committed_text)
        except Exception as e:
            print(f"[stream] live type failed: {e}")

    def restore_clipboard(self):
        if self.saved_clip is not None:
            try:
                pyperclip.copy(self.saved_clip)
            except Exception:
                pass
            self.saved_clip = None


class Overlay:
    """Small always-on-top pill showing recording state (red = listening,
    amber = transcribing), bottom-center of the screen. Tk lives in its own
    thread; state changes arrive via a queue."""

    def __init__(self):
        import queue
        self._q = queue.Queue()
        self._empty = queue.Empty
        threading.Thread(target=self._run, daemon=True).start()

    def set(self, state):
        self._q.put(state)

    def _run(self):
        try:
            self._build_and_loop()
        except Exception as e:
            # An overlay failure must NEVER take down dictation — just disable it.
            print(f"[overlay] disabled after error: {e}")

    def _build_and_loop(self):
        try:
            import tkinter as tk
        except Exception as e:
            print(f"[overlay] unavailable: {e}")
            return
        bg = "#15171c"
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.93)
        except Exception:
            pass
        root.configure(bg=bg)
        canvas = tk.Canvas(root, width=14, height=14, bg=bg, highlightthickness=0)
        dot = canvas.create_oval(2, 2, 12, 12, fill="#dc3c3c", outline="")
        canvas.pack(side="left", padx=(12, 6), pady=9)
        label = tk.Label(root, text="Listening", fg="#e6e6e6", bg=bg,
                         font=("Segoe UI", 10), wraplength=420, justify="left")
        label.pack(side="left", padx=(0, 14))
        hide_at = [None]
        root.update_idletasks()

        def place():
            # Bottom-center of the ACTIVE window — follows whichever monitor
            # and app you're dictating into.
            w = root.winfo_reqwidth()
            try:
                import ctypes
                from ctypes import wintypes
                h = ctypes.windll.user32.GetForegroundWindow()
                r = wintypes.RECT()
                if h and ctypes.windll.user32.GetWindowRect(h, ctypes.byref(r)) \
                        and r.right - r.left > 200 and r.bottom - r.top > 150:
                    x = (r.left + r.right) // 2 - w // 2
                    y = r.bottom - 90
                    root.geometry(f"+{x}+{y}")
                    return
            except Exception:
                pass
            x = (root.winfo_screenwidth() - w) // 2
            root.geometry(f"+{x}+{root.winfo_screenheight() - 110}")

        def poll():
            try:
                while True:
                    state = self._q.get_nowait()
                    if isinstance(state, tuple) and state[0] == "say":
                        # Command response: show the text so command mode never
                        # depends on audio alone.
                        canvas.itemconfig(dot, fill="#3ec97e")
                        label.config(text=state[1][:300])
                        place()
                        root.deiconify()
                        hide_at[0] = time.time() + max(4, min(12, len(state[1]) / 18))
                    elif state == "rec":
                        canvas.itemconfig(dot, fill="#dc3c3c")
                        label.config(text="Listening")
                        hide_at[0] = None
                        place()
                        root.deiconify()
                    elif state == "work":
                        canvas.itemconfig(dot, fill="#f0b428")
                        label.config(text="Transcribing…")
                        hide_at[0] = None
                        root.deiconify()
                    else:
                        if hide_at[0] is None:
                            root.withdraw()
            except self._empty:
                pass
            except Exception as e:
                print(f"[overlay] poll error (continuing): {e}")
            if hide_at[0] is not None and time.time() > hide_at[0]:
                hide_at[0] = None
                root.withdraw()
            root.after(80, poll)  # always reschedule — never let the loop die

        poll()
        root.mainloop()


class OutputMuter:
    """Mutes the default speaker endpoint during recording, restoring prior state."""

    def __init__(self):
        self._was_muted = None

    def _endpoint(self):
        from pycaw.pycaw import AudioUtilities
        return AudioUtilities.GetSpeakers().EndpointVolume

    def mute(self):
        try:
            import comtypes
            comtypes.CoInitialize()
            vol = self._endpoint()
            self._was_muted = bool(vol.GetMute())
            vol.SetMute(1, None)
        except Exception as e:
            print(f"[mute] failed: {e}")
            self._was_muted = None

    def restore(self):
        if self._was_muted is None:
            return
        try:
            self._endpoint().SetMute(1 if self._was_muted else 0, None)
        except Exception as e:
            print(f"[mute] restore failed: {e}")
        self._was_muted = None

    def ensure_unmuted(self):
        """Startup recovery: a crash mid-recording leaves speakers muted."""
        try:
            import comtypes
            comtypes.CoInitialize()
            vol = self._endpoint()
            if vol.GetMute():
                vol.SetMute(0, None)
                print("[mute] speakers were left muted (crashed session?) — unmuted")
        except Exception:
            pass


def llm_cleanup(text, cfg, window=None):
    c = cfg["cleanup"]
    style = match_profile(cfg, window)
    if style is None and c.get("mode", "matched") == "matched":
        return text  # unmatched app -> raw fast path, no LLM call
    if len(text.split()) < c.get("min_words", 5):
        return text  # short snippets: delay is felt most, styling worth least
    system = CLEANUP_PROMPT
    if style:
        system += f"\nTarget style for this app ({window}): {style}"
    t0 = time.time()
    out, lane = llm.chain(c, system, text)
    if out is None:
        print("[cleanup] all providers failed, using raw transcript")
        return text
    print(f"[cleanup] {'styled:' + window if style else 'neutral'} "
          f"via {lane} in {time.time() - t0:.2f}s")
    return out


def foreground_window():
    try:
        import ctypes
        h = ctypes.windll.user32.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetWindowTextW(h, buf, 256)
        return buf.value
    except Exception:
        return "?"


def paste_text(text, mode, restore=True):
    print(f"[paste] mode={mode} target window: {foreground_window()!r}")
    if mode == "type":
        keyboard.write(text)
        return
    old = None
    if restore:
        try:
            old = pyperclip.paste()
        except Exception:
            pass
    pyperclip.copy(text)
    time.sleep(0.02)
    keyboard.press("ctrl")
    keyboard.press("v")
    time.sleep(0.03)
    keyboard.release("v")
    keyboard.release("ctrl")
    print("[paste] ctrl+v sent")
    if old is not None:
        def restore():
            time.sleep(0.6)
            try:
                pyperclip.copy(old)
            except Exception:
                pass
        threading.Thread(target=restore, daemon=True).start()


class Flow:
    def __init__(self):
        self.cfg = load_config()
        self.recorder = Recorder(self.cfg["sample_rate"],
                                 pre_roll_s=self.cfg.get("pre_roll_s", 0.4))
        self.muter = OutputMuter()
        self.muter.ensure_unmuted()
        self.recording = False
        self.active_mode = "dictate"
        self.streamer = None
        self._key_down = False
        import queue as _queue
        self._ctrl_q = _queue.Queue()
        self.busy = threading.Lock()
        self.model_lock = threading.Lock()  # serialize GPU across hotkey + server
        self.router = Router(self.cfg)
        threading.Thread(target=self._ctrl_loop, daemon=True,
                         name="hotkey-ctrl").start()
        self.started_at = time.time()
        self.last_pull = None  # last delta sync by multivac (server sets this)
        self.tray = None
        self.overlay = Overlay() if self.cfg.get("overlay", True) else None
        h = self.cfg["history"]
        self.store = TranscriptStore(
            APP_DIR / h["dir"], archive_audio=h["archive_audio"]
        ) if h["enabled"] else None
        print(f"Loading {self.cfg['model']} on {self.cfg['device']}...")
        t0 = time.time()
        self.model = WhisperModel(
            self.cfg["model"], device=self.cfg["device"],
            compute_type=self.cfg["compute_type"],
        )
        # Warm up so the first real dictation isn't slow.
        self.model.transcribe(np.zeros(self.cfg["sample_rate"], dtype=np.float32))
        if self.cfg["cleanup"]["enabled"] and self.cfg["cleanup"]["provider"] == "ollama":
            threading.Thread(target=self._warm_ollama, daemon=True).start()
        print(f"Model ready in {time.time() - t0:.1f}s. "
              f"Hold [{self.cfg['hotkey']}] to dictate, "
              f"[{self.cfg.get('command_hotkey')}] for a command.")

    def _warm_ollama(self):
        import requests
        c = self.cfg["cleanup"]
        try:
            t0 = time.time()
            requests.post(
                (c["base_url"] or "http://127.0.0.1:11434") + "/api/chat",
                json={"model": c["model"], "stream": False, "think": False,
                      "keep_alive": -1,
                      "messages": [{"role": "user", "content": "hi"}]},
                timeout=300,
            ).raise_for_status()
            print(f"[cleanup] local model {c['model']} warm in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[cleanup] warmup failed ({e}) — is Ollama running?")

    def set_icon(self, state):
        if self.tray:
            self.tray.icon = make_icon(state)
        if self.overlay:
            self.overlay.set(state)

    def on_press(self, mode="dictate"):
        if self.recording or self.busy.locked():
            return
        self.active_mode = mode
        self.recording = True
        self.recorder.start()  # first — capture must not wait on mute/icon
        self.set_icon("rec")
        if self.cfg.get("mute_output_while_recording"):
            self.muter.mute()
        if (mode == "dictate" and self.cfg["streaming"]["enabled"]):
            # Live typing only where no style profile applies — styled text
            # must wait for the whole utterance.
            live = (self.cfg["streaming"].get("live_type", True)
                    and not (self.cfg["cleanup"]["enabled"]
                             and match_profile(self.cfg, foreground_window())))
            self.streamer = StreamWorker(self, live=live)
            self.streamer.start()

    def on_release(self):
        if not self.recording:
            return
        self.recording = False
        streamer = self.streamer
        self.streamer = None
        if streamer:
            streamer.stop()
        audio = self.recorder.stop()
        self.muter.restore()
        if self.active_mode == "command":
            threading.Thread(target=self.process_command, args=(audio,),
                             daemon=True).start()
        else:
            threading.Thread(target=self.process, args=(audio, streamer),
                             daemon=True).start()

    def bias_prompt(self, committed=""):
        """initial_prompt = personal vocabulary + committed text so far."""
        vocab = self.cfg.get("vocab") or []
        parts = []
        if vocab:
            parts.append(", ".join(vocab) + ".")
        if committed:
            parts.append(committed[-200:])
        return " ".join(parts) or None

    def decode(self, audio, beam_size=None, prompt=None, word_timestamps=False):
        """One locked whisper pass -> list of segments."""
        vad_params = ({
            "threshold": self.cfg["vad_threshold"],
            "min_silence_duration_ms": self.cfg.get("vad_min_silence_ms", 400),
            "speech_pad_ms": self.cfg.get("vad_speech_pad_ms", 600),
        } if self.cfg.get("vad_filter") else None)
        with self.model_lock:
            segments, _ = self.model.transcribe(
                audio, language=self.cfg["language"],
                vad_filter=self.cfg.get("vad_filter", True),
                vad_parameters=vad_params,
                beam_size=beam_size or self.cfg.get("beam_size", 5),
                condition_on_previous_text=self.cfg.get("condition_on_previous_text", False),
                no_speech_threshold=self.cfg.get("no_speech_threshold"),
                initial_prompt=prompt,
                word_timestamps=word_timestamps,
            )
            return list(segments)

    def transcribe_and_store(self, audio, sample_rate=None, source="pc-hotkey",
                             window=None, committed=("", 0), skip_cleanup=False):
        """Shared pipeline: resample -> ASR (tail only if streaming committed a
        prefix) -> cleanup -> store. Returns (text, meta)."""
        target_sr = self.cfg["sample_rate"]
        if sample_rate and sample_rate != target_sr:
            audio = resample_linear(audio, sample_rate, target_sr)
        dur = len(audio) / target_sr
        if self.cfg.get("normalize_gain"):
            peak = float(np.abs(audio).max()) or 1.0
            audio = (audio / peak * 0.95).astype(np.float32)
        committed_text, committed_samples = committed
        t0 = time.time()
        tail = audio[committed_samples:]
        segs = self.decode(tail, prompt=self.bias_prompt(committed_text)) \
            if len(tail) > 0 else []
        raw = (committed_text + " "
               + " ".join(s.text.strip() for s in segs)).strip()
        asr_t = time.time() - t0
        text = raw
        if raw and self.cfg["cleanup"]["enabled"] and not skip_cleanup:
            text = llm_cleanup(raw, self.cfg, window=window)
        if self.store and raw:
            self.store.add(raw, source=source, window=window, duration_s=dur,
                           asr_s=asr_t, model=self.cfg["model"],
                           device=self.cfg["device"], audio=audio,
                           sample_rate=target_sr, cleaned=text)
        print(f"[{source}] [{dur:.1f}s audio -> {asr_t:.2f}s asr] {text}")
        return text, {"asr_s": round(asr_t, 2), "duration_s": round(dur, 2)}

    def process(self, audio, streamer=None):
        with self.busy:
            self.set_icon("work")
            try:
                dur = len(audio) / self.cfg["sample_rate"]
                if dur < self.cfg["min_duration_s"]:
                    return
                if self.cfg.get("save_last_audio"):
                    save_wav(APP_DIR / "last.wav", audio, self.cfg["sample_rate"])
                committed = ((streamer.committed_text, streamer.committed_samples)
                             if streamer else ("", 0))
                live_typed = streamer.typed_len if streamer else 0
                text, _ = self.transcribe_and_store(
                    audio, source="pc-hotkey", window=foreground_window(),
                    committed=committed, skip_cleanup=live_typed > 0)
                if text and live_typed:
                    # Part of the text is already on screen — flush the rest.
                    remainder = apply_replacements(text[live_typed:], self.cfg)
                    if remainder:
                        paste_text(remainder, "clipboard", restore=False)
                elif text:
                    paste_text(apply_replacements(text, self.cfg),
                               self.cfg["paste_mode"])
            finally:
                if streamer:
                    time.sleep(0.3)
                    streamer.restore_clipboard()
                self.set_icon("idle")

    def process_command(self, audio):
        with self.busy:
            self.set_icon("work")
            try:
                dur = len(audio) / self.cfg["sample_rate"]
                if dur < self.cfg["min_duration_s"]:
                    return
                text, _ = self.transcribe_and_store(
                    audio, source="pc-command", window=foreground_window())
                if not text:
                    return
                print(f"[command] heard: {text}")
                response = self.router.handle(text)
                print(f"[command] -> {response}")
                if self.overlay and response:
                    self.overlay.set(("say", response))
                speak(response)
            finally:
                self.set_icon("idle")

    def _on_key(self, event):
        # This runs inside the keyboard listener thread — it MUST be trivial and
        # never throw, or the listener dies and every hotkey stops working until
        # restart. So: match, drop auto-repeat, enqueue. All real work (mute,
        # audio start, transcription) happens in _ctrl_loop.
        if event.name == self.cfg["hotkey"]:
            mode = "dictate"
        elif event.name == self.cfg.get("command_hotkey"):
            mode = "command"
        else:
            return
        if event.event_type == keyboard.KEY_DOWN:
            if self._key_down:
                return  # ignore held-key auto-repeat
            self._key_down = True
            self._ctrl_q.put(("down", mode))
        elif event.event_type == keyboard.KEY_UP:
            if not self._key_down:
                return
            self._key_down = False
            self._ctrl_q.put(("up", mode))

    def _ctrl_loop(self):
        while True:
            action, mode = self._ctrl_q.get()
            try:
                if action == "down":
                    self.on_press(mode)
                else:
                    self.on_release()
            except Exception as e:
                # Never let a press/release error propagate — recover the state.
                print(f"[hotkey] {action} handler error: {e}")
                self.recording = False

    def maybe_start_server(self):
        s = self.cfg["server"]
        if not s["enabled"]:
            return
        if not s["token"]:
            import secrets
            s["token"] = secrets.token_urlsafe(16)
            _persist_config({"server": {"token": s["token"]}})
            print(f"[server] generated token: {s['token']}")
        from server import make_server
        httpd = make_server(self, s["host"], s["port"], s["token"])
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"[server] listening on {s['host']}:{s['port']} "
              f"(token required; history page at /)")

    def run(self):
        self.maybe_start_server()
        # Raw hook (never suppresses) so we can distinguish left vs right ctrl
        # and never eat the key from other apps.
        keyboard.hook(self._on_key)
        try:
            import webbrowser

            import pystray
            from pystray import MenuItem
            port = self.cfg["server"]["port"]
            self.tray = pystray.Icon(
                "flow", make_icon("idle"), "Flow — hold %s to dictate" % self.cfg["hotkey"],
                menu=pystray.Menu(
                    MenuItem("Open history page",
                             lambda: webbrowser.open(
                                 f"http://127.0.0.1:{port}/#tok="
                                 + self.cfg["server"]["token"])),
                    MenuItem("Edit config (restart to apply)",
                             lambda: os.startfile(str(CONFIG_PATH))),
                    MenuItem("Open data folder",
                             lambda: os.startfile(str(APP_DIR / "data"))),
                    MenuItem("Quit", lambda: self.tray.stop()),
                ),
            )
            self.tray.run()  # blocks until Quit
        except Exception as e:
            print(f"[tray] unavailable ({e}); running headless. Ctrl+C to quit.")
            keyboard.wait()


def make_icon(state):
    from PIL import Image, ImageDraw
    color = {"idle": (90, 90, 90), "rec": (220, 60, 60), "work": (240, 180, 40)}[state]
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 56, 56], fill=color)
    d.rounded_rectangle([26, 16, 38, 40], radius=6, fill=(255, 255, 255))
    d.line([32, 44, 32, 50], fill=(255, 255, 255), width=4)
    return img


def cli_history(limit=30):
    cfg = load_config()
    store = TranscriptStore(APP_DIR / cfg["history"]["dir"])
    entries = store.read(limit=limit)
    if not entries:
        print("No dictations recorded yet.")
        return
    for e in entries:
        print(f"{e['ts']}  [{e.get('source','?'):9}] {e['text']}")
    s = store.stats()
    print(f"\n{s['dictations']} dictations · {s['words']} words · {s['by_source']}")


def acquire_single_instance():
    import socket
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 18756))
        return s  # held for process lifetime
    except OSError:
        print("Flow is already running — exiting.")
        sys.exit(0)  # clean, so the supervisor doesn't treat it as a crash


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--history", "-H"):
        cli_history(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif len(sys.argv) > 1 and sys.argv[1] in ("--stats", "-s"):
        cfg = load_config()
        print(TranscriptStore(APP_DIR / cfg["history"]["dir"]).stats())
    else:
        _lock = acquire_single_instance()
        Flow().run()
