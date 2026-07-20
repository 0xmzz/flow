"""Local text-to-speech with a neural Kokoro lane and SAPI fallback."""

from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import numpy as np


MODEL_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
             "model-files-v1.0/kokoro-v1.0.onnx")
VOICES_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
              "model-files-v1.0/voices-v1.0.bin")
MODEL_ROOT = Path(__file__).parent / "data" / "models" / "kokoro"

_cfg = {"engine": "sapi", "voice": "af_heart", "speed": 1.05}
_engine = None
_provider = "unavailable"
_warm_state = "cold"
_warm_lock = threading.Lock()
_speak_lock = threading.Lock()
_cancel_lock = threading.Lock()
_cancel_generation = 0
_play = _wait = _stop_playback = None


def _sapi(text):
    import comtypes
    import comtypes.client
    comtypes.CoInitialize()
    comtypes.client.CreateObject("SAPI.SpVoice").Speak(text)


def _download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, destination)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _ensure_assets(root):
    model = root / "kokoro-v1.0.onnx"
    voices = root / "voices-v1.0.bin"
    if not model.is_file():
        print(f"[tts] downloading Kokoro model to {root}")
        _download(MODEL_URL, model)
    if not voices.is_file():
        print(f"[tts] downloading Kokoro voices to {root}")
        _download(VOICES_URL, voices)
    return model, voices


def _make_engine(root=MODEL_ROOT):
    from live import ParakeetEngine
    ParakeetEngine._preload_gpu_dlls()

    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    model, voices = _ensure_assets(Path(root))
    try:
        session = ort.InferenceSession(
            str(model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("CUDA provider fell back to CPU")
        provider = "CUDA"
    except Exception as exc:
        print(f"[tts] CUDA Kokoro failed ({exc}); falling back to CPU")
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        provider = "CPU"
    return Kokoro.from_session(session, str(voices)), provider


def configure(cfg, *, start=True):
    """Apply TTS config and optionally begin one background warmup."""
    global _cfg, _warm_state, _engine
    requested = dict(cfg or {})
    engine = str(requested.get("engine", "sapi")).lower()
    requested["engine"] = engine if engine in {"kokoro", "sapi"} else "sapi"
    with _warm_lock:
        _cfg = {"engine": requested["engine"],
                "voice": requested.get("voice", "af_heart"),
                "speed": float(requested.get("speed", 1.05))}
        _engine = None
        _warm_state = "cold"
    if start and _cfg["engine"] == "kokoro":
        warmup()


def warmup():
    """Start loading Kokoro once; speech uses SAPI until it is ready."""
    global _warm_state
    with _warm_lock:
        if _cfg["engine"] != "kokoro" or _warm_state != "cold":
            return
        _warm_state = "warming"
    threading.Thread(target=_warm, daemon=True, name="kokoro-warmup").start()


def _warm():
    global _engine, _provider, _warm_state
    try:
        engine, provider = _make_engine()
        engine.create("Ready.", voice=_cfg["voice"], speed=_cfg["speed"],
                      lang="en-us")
        with _warm_lock:
            _engine, _provider, _warm_state = engine, provider, "ready"
        print(f"[tts] Kokoro warm; provider={provider}")
    except Exception as exc:
        with _warm_lock:
            _engine, _provider, _warm_state = None, "unavailable", "failed"
        print(f"[tts] Kokoro warmup failed ({exc}); using SAPI")


def _audio_functions():
    global _play, _wait, _stop_playback
    if _play is None:
        import sounddevice as sd
        _play, _wait, _stop_playback = sd.play, sd.wait, sd.stop
    return _play, _wait


def speak(text):
    """Speak one utterance synchronously; concurrent calls are serialized."""
    if not text:
        return
    with _cancel_lock:
        generation = _cancel_generation
    with _speak_lock:
        with _cancel_lock:
            if generation != _cancel_generation:
                return
        with _warm_lock:
            engine = _engine if (_cfg["engine"] == "kokoro"
                                  and _warm_state == "ready") else None
            voice, speed = _cfg["voice"], _cfg["speed"]
        if engine is None:
            try:
                _sapi(str(text))
            except Exception as exc:
                print(f"[speak] {exc}")
            return
        try:
            samples, rate = engine.create(str(text), voice=voice, speed=speed,
                                          lang="en-us")
        except Exception as exc:
            print(f"[tts] synthesis failed ({exc}); using SAPI")
            try:
                _sapi(str(text))
            except Exception as sapi_exc:
                print(f"[speak] {sapi_exc}")
            return
        try:
            play, wait = _audio_functions()
            play(np.asarray(samples, dtype=np.float32).reshape(-1),
                 samplerate=rate, device=None)
            wait()
        except Exception as exc:
            print(f"[tts] playback failed: {exc}")


def stop():
    """Abort current playback and invalidate speech queued behind it."""
    global _cancel_generation
    with _cancel_lock:
        _cancel_generation += 1
    stopper = _stop_playback
    if stopper is None:
        try:
            import sounddevice as sd
            stopper = sd.stop
        except Exception:
            return
    try:
        stopper()
    except Exception as exc:
        print(f"[tts] stop failed: {exc}")

