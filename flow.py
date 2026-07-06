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

from store import TranscriptStore, resample_linear, save_wav

APP_DIR = Path(__file__).parent
CONFIG_PATH = APP_DIR / "config.json"

DEFAULTS = {
    "hotkey": "f8",                # hold to record, release to transcribe
    "model": "large-v3-turbo",     # any faster-whisper model id
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
    },
    "paste_mode": "clipboard",     # "clipboard" (fast, restores clipboard) or "type"
    "mute_output_while_recording": True,  # silence speakers during dictation
    "min_duration_s": 0.3,
    "save_last_audio": False,      # debug: write each recording to last.wav for tuning
    # ASR tuning — lower vad thresholds and gain help quiet/whispered speech.
    "vad_filter": True,
    "vad_threshold": 0.5,          # lower = keeps quieter speech (0.1-0.3 for whisper)
    "beam_size": 5,
    "normalize_gain": False,       # peak-normalize audio before ASR
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
    "errors from context. Do NOT rephrase, summarize, or add anything. Preserve the "
    "speaker's wording and meaning exactly. Reply with only the cleaned text."
)


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

    def stop(self):
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).flatten()


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


def llm_cleanup(text, cfg):
    import requests

    c = cfg["cleanup"]
    key = os.environ.get(c["api_key_env"], "")
    if not key:
        return text
    try:
        if c["provider"] == "anthropic":
            r = requests.post(
                (c["base_url"] or "https://api.anthropic.com") + "/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={
                    "model": c["model"],
                    "max_tokens": 2048,
                    "system": CLEANUP_PROMPT,
                    "messages": [{"role": "user", "content": text}],
                },
                timeout=15,
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"].strip()
        else:  # openai-compatible
            r = requests.post(
                (c["base_url"] or "https://api.openai.com/v1") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": c["model"],
                    "messages": [
                        {"role": "system", "content": CLEANUP_PROMPT},
                        {"role": "user", "content": text},
                    ],
                },
                timeout=15,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[cleanup] failed, using raw transcript: {e}")
        return text


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
    time.sleep(0.15)
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
        self.busy = threading.Lock()
        self.model_lock = threading.Lock()  # serialize GPU across hotkey + server
        self.tray = None
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
        print(f"Model ready in {time.time() - t0:.1f}s. "
              f"Hold [{self.cfg['hotkey']}] to dictate.")

    def set_icon(self, state):
        if self.tray:
            self.tray.icon = make_icon(state)

    def on_press(self):
        if self.recording or self.busy.locked():
            return
        self.recording = True
        self.set_icon("rec")
        if self.cfg.get("mute_output_while_recording"):
            self.muter.mute()
        self.recorder.start()

    def on_release(self):
        if not self.recording:
            return
        self.recording = False
        audio = self.recorder.stop()
        self.muter.restore()
        threading.Thread(target=self.process, args=(audio,), daemon=True).start()

    def transcribe_and_store(self, audio, sample_rate=None, source="pc-hotkey",
                             window=None):
        """Shared pipeline: resample -> ASR -> cleanup -> store. Returns (text, meta)."""
        target_sr = self.cfg["sample_rate"]
        if sample_rate and sample_rate != target_sr:
            audio = resample_linear(audio, sample_rate, target_sr)
        dur = len(audio) / target_sr
        if self.cfg.get("normalize_gain"):
            peak = float(np.abs(audio).max()) or 1.0
            audio = (audio / peak * 0.95).astype(np.float32)
        vad_params = ({"threshold": self.cfg["vad_threshold"]}
                      if self.cfg.get("vad_filter") else None)
        t0 = time.time()
        with self.model_lock:
            segments, _ = self.model.transcribe(
                audio, language=self.cfg["language"],
                vad_filter=self.cfg.get("vad_filter", True),
                vad_parameters=vad_params,
                beam_size=self.cfg.get("beam_size", 5),
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        asr_t = time.time() - t0
        if text and self.cfg["cleanup"]["enabled"]:
            text = llm_cleanup(text, self.cfg)
        if self.store and text:
            self.store.add(text, source=source, window=window, duration_s=dur,
                           asr_s=asr_t, model=self.cfg["model"],
                           device=self.cfg["device"], audio=audio,
                           sample_rate=target_sr)
        print(f"[{source}] [{dur:.1f}s audio -> {asr_t:.2f}s asr] {text}")
        return text, {"asr_s": round(asr_t, 2), "duration_s": round(dur, 2)}

    def process(self, audio):
        with self.busy:
            self.set_icon("work")
            try:
                dur = len(audio) / self.cfg["sample_rate"]
                if dur < self.cfg["min_duration_s"]:
                    return
                if self.cfg.get("save_last_audio"):
                    save_wav(APP_DIR / "last.wav", audio, self.cfg["sample_rate"])
                text, _ = self.transcribe_and_store(
                    audio, source="pc-hotkey", window=foreground_window())
                if text:
                    paste_text(text, self.cfg["paste_mode"])
            finally:
                self.set_icon("idle")

    def _on_key(self, event):
        # Match the exact key NAME ("right ctrl"), not the scan code — left and
        # right ctrl share scan code 29, so on_press_key fires for both.
        if event.name != self.cfg["hotkey"]:
            return
        if event.event_type == keyboard.KEY_DOWN:
            self.on_press()
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
            import pystray
            from pystray import MenuItem
            self.tray = pystray.Icon(
                "flow", make_icon("idle"), "Flow — hold %s to dictate" % self.cfg["hotkey"],
                menu=pystray.Menu(
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
