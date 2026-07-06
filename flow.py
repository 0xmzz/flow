"""Flow — local push-to-talk dictation for Windows.

Hold the hotkey, speak, release: transcription is typed at your cursor.
ASR runs fully local on GPU (faster-whisper). Optional LLM cleanup pass.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

# Expose pip-installed NVIDIA cuBLAS/cuDNN DLLs to ctranslate2 before import.
_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if _nvidia.is_dir():
    for _bin in _nvidia.glob("*/bin"):
        os.add_dll_directory(str(_bin))
        os.environ["PATH"] = str(_bin) + os.pathsep + os.environ["PATH"]

if sys.stdout:
    sys.stdout.reconfigure(line_buffering=True)

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
    "min_duration_s": 0.3,
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
    "beam_size": 5,
    "normalize_gain": False,       # peak-normalize before ASR (low value; A/B only)
    "streaming": {
        "enabled": True,           # transcribe while the hotkey is held
        "interval_s": 1.2,         # rolling-pass cadence
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
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self._chunks = []
        self._stream = None

    def start(self):
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            callback=lambda indata, *_: self._chunks.append(indata.copy()),
        )
        self._stream.start()

    def snapshot(self):
        """Audio captured so far (recording continues)."""
        chunks = list(self._chunks)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).flatten()

    def stop(self):
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).flatten()


class StreamWorker:
    """Transcribes rolling chunks WHILE the hotkey is held, committing text that
    is stable across two consecutive passes (LocalAgreement). At release only
    the uncommitted tail needs decoding, so release-to-paste stays ~constant no
    matter how long the dictation ran."""

    # Segments ending within this many seconds of the buffer edge are still
    # "hot" (may change on the next pass) and are never committed.
    UNSTABLE_TAIL_S = 1.0

    def __init__(self, flow):
        self.flow = flow
        self.committed_text = ""
        self.committed_samples = 0
        self._prev = []          # segment texts from the previous pass
        self._active = False
        self._thread = None

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
        interval = self.flow.cfg["streaming"].get("interval_s", 1.2)
        while self._active:
            time.sleep(interval)
            if not self._active:
                return
            snap = self.flow.recorder.snapshot()
            tail = snap[self.committed_samples:]
            if len(tail) < sr * 1.5:
                continue
            try:
                segs = self.flow.decode(tail, beam_size=1, word_timestamps=True,
                                        prompt=self.committed_text[-200:] or None)
            except Exception as e:
                print(f"[stream] pass failed: {e}")
                continue
            if not self._active:   # released mid-pass: let the final pass own it
                return
            tail_dur = len(tail) / sr
            # Word-level commit: whisper often yields ONE growing segment, so
            # segment-level stability never triggers on normal sentences.
            words = [w for s in segs for w in (s.words or [])
                     if w.end < tail_dur - self.UNSTABLE_TAIL_S]
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
                if nxt - words[i].end >= 0.35:
                    cut = i + 1
            if cut:
                phrase = "".join(w.word for w in words[:cut]).strip()
                self.committed_text = (self.committed_text + " " + phrase).strip()
                self.committed_samples += int(words[cut - 1].end * sr)
                self._prev = texts[cut:]
            else:
                self._prev = texts


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
                         font=("Segoe UI", 10))
        label.pack(side="left", padx=(0, 14))
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
                    if state == "rec":
                        canvas.itemconfig(dot, fill="#dc3c3c")
                        label.config(text="Listening")
                        place()
                        root.deiconify()
                    elif state == "work":
                        canvas.itemconfig(dot, fill="#f0b428")
                        label.config(text="Transcribing…")
                        root.deiconify()
                    else:
                        root.withdraw()
            except self._empty:
                pass
            root.after(80, poll)

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


def paste_text(text, mode):
    print(f"[paste] mode={mode} target window: {foreground_window()!r}")
    if mode == "type":
        keyboard.write(text)
        return
    old = None
    try:
        old = pyperclip.paste()
    except Exception:
        pass
    pyperclip.copy(text)
    time.sleep(0.05)
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
        self.recorder = Recorder(self.cfg["sample_rate"])
        self.muter = OutputMuter()
        self.recording = False
        self.active_mode = "dictate"
        self.streamer = None
        self.busy = threading.Lock()
        self.model_lock = threading.Lock()  # serialize GPU across hotkey + server
        self.router = Router(self.cfg)
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
        self.set_icon("rec")
        if self.cfg.get("mute_output_while_recording"):
            self.muter.mute()
        self.recorder.start()
        if (mode == "dictate" and self.cfg["streaming"]["enabled"]):
            self.streamer = StreamWorker(self)
            self.streamer.start()

    def on_release(self):
        if not self.recording:
            return
        self.recording = False
        committed = ("", 0)
        if self.streamer:
            committed = self.streamer.stop()
            self.streamer = None
        audio = self.recorder.stop()
        self.muter.restore()
        if self.active_mode == "command":
            threading.Thread(target=self.process_command, args=(audio,),
                             daemon=True).start()
        else:
            threading.Thread(target=self.process, args=(audio, committed),
                             daemon=True).start()

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
                             window=None, committed=("", 0)):
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
        segs = self.decode(tail, prompt=committed_text[-200:] or None) \
            if len(tail) > 0 else []
        raw = (committed_text + " "
               + " ".join(s.text.strip() for s in segs)).strip()
        asr_t = time.time() - t0
        text = raw
        if raw and self.cfg["cleanup"]["enabled"]:
            text = llm_cleanup(raw, self.cfg, window=window)
        if self.store and raw:
            self.store.add(raw, source=source, window=window, duration_s=dur,
                           asr_s=asr_t, model=self.cfg["model"],
                           device=self.cfg["device"], audio=audio,
                           sample_rate=target_sr, cleaned=text)
        print(f"[{source}] [{dur:.1f}s audio -> {asr_t:.2f}s asr] {text}")
        return text, {"asr_s": round(asr_t, 2), "duration_s": round(dur, 2)}

    def process(self, audio, committed=("", 0)):
        with self.busy:
            self.set_icon("work")
            try:
                dur = len(audio) / self.cfg["sample_rate"]
                if dur < self.cfg["min_duration_s"]:
                    return
                if self.cfg.get("save_last_audio"):
                    save_wav(APP_DIR / "last.wav", audio, self.cfg["sample_rate"])
                text, _ = self.transcribe_and_store(
                    audio, source="pc-hotkey", window=foreground_window(),
                    committed=committed)
                if text:
                    paste_text(text, self.cfg["paste_mode"])
            finally:
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
                speak(response)
            finally:
                self.set_icon("idle")

    def _on_key(self, event):
        # Match the exact key NAME, not the scan code — left and right ctrl share
        # scan code 29, so matching by name keeps the two hotkeys distinct.
        if event.name == self.cfg["hotkey"]:
            mode = "dictate"
        elif event.name == self.cfg.get("command_hotkey"):
            mode = "command"
        else:
            return
        if event.event_type == keyboard.KEY_DOWN:
            self.on_press(mode)
        elif event.event_type == keyboard.KEY_UP:
            self.on_release()

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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--history", "-H"):
        cli_history(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif len(sys.argv) > 1 and sys.argv[1] in ("--stats", "-s"):
        cfg = load_config()
        print(TranscriptStore(APP_DIR / cfg["history"]["dir"]).stats())
    else:
        Flow().run()
