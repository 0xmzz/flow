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

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel

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
    "min_duration_s": 0.3,
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
        user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    else:
        CONFIG_PATH.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
    return cfg


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


def paste_text(text, mode):
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
    keyboard.send("ctrl+v")
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
        self.recording = False
        self.busy = threading.Lock()
        self.tray = None
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
        self.recorder.start()

    def on_release(self):
        if not self.recording:
            return
        self.recording = False
        audio = self.recorder.stop()
        threading.Thread(target=self.process, args=(audio,), daemon=True).start()

    def process(self, audio):
        with self.busy:
            self.set_icon("work")
            try:
                dur = len(audio) / self.cfg["sample_rate"]
                if dur < self.cfg["min_duration_s"]:
                    return
                t0 = time.time()
                segments, _ = self.model.transcribe(
                    audio, language=self.cfg["language"],
                    vad_filter=True, beam_size=5,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                asr_t = time.time() - t0
                if not text:
                    return
                if self.cfg["cleanup"]["enabled"]:
                    text = llm_cleanup(text, self.cfg)
                paste_text(text, self.cfg["paste_mode"])
                print(f"[{dur:.1f}s audio -> {asr_t:.2f}s asr] {text}")
            finally:
                self.set_icon("idle")

    def run(self):
        keyboard.on_press_key(self.cfg["hotkey"], lambda _: self.on_press(),
                              suppress=True)
        keyboard.on_release_key(self.cfg["hotkey"], lambda _: self.on_release(),
                                suppress=True)
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


if __name__ == "__main__":
    Flow().run()
