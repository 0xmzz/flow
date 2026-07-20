"""Always-listening local voice-command mode for Flow."""

from __future__ import annotations

import ctypes
import difflib
import glob
import os
import queue
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512                 # 32 ms, required by Silero
PRE_FRAMES = 5                      # 160 ms
POST_FRAMES = 5                     # 160 ms
GREETINGS = {"hey", "hi", "ok", "okay", "yo", "so"}


def wake_command(text, wake_word="flow"):
    """Return command, empty string for wake-only, or None for no wake match."""
    text = (text or "").strip()
    if not text:
        return None
    if wake_word is None:
        return text
    parts = text.split()
    first = re.sub(r"[^\w]", "", parts[0].lower(), flags=re.UNICODE)
    if first in GREETINGS and len(parts) > 1:
        parts = parts[1:]
        first = re.sub(r"[^\w]", "", parts[0].lower(), flags=re.UNICODE)
    wake = re.sub(r"[^\w]", "", str(wake_word).lower(), flags=re.UNICODE)
    if not first or not wake or difflib.SequenceMatcher(None, first, wake).ratio() < 0.75:
        return None
    return " ".join(parts[1:]).strip()


class SileroVad:
    """Tiny no-Torch streaming Silero wrapper."""

    def __init__(self):
        from pysilero_vad import SileroVoiceActivityDetector
        self._vad = SileroVoiceActivityDetector()

    def reset(self):
        self._vad.reset()

    def probability(self, frame):
        return float(self._vad.process_samples(frame.tolist()))


class ParakeetEngine:
    """Parakeet via onnx-asr, with CUDA warmup and CPU-int8 fallback."""

    def __init__(self, model_name, model_root):
        self.model_name = model_name
        self.model_root = Path(model_root)
        self.model_root.parent.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.model_root.parent / "huggingface"))
        os.environ.setdefault("HF_HUB_CACHE", str(self.model_root.parent / "huggingface" / "hub"))
        self.model = None
        self.provider = "unavailable"
        self._load()

    @staticmethod
    def _preload_gpu_dlls():
        # ORT loads onnxruntime_providers_cuda.dll with its own search flags, so
        # add_dll_directory/preload_dlls miss the pip nvidia-* wheels; loading
        # each DLL by absolute path makes the loader reuse them by name.
        dlls = glob.glob(os.path.join(sys.prefix, "Lib", "site-packages",
                                      "nvidia", "*", "bin", "*.dll"))
        for _ in range(3):  # passes so interdependent DLLs land in any order
            for p in dlls:
                try:
                    ctypes.WinDLL(os.path.abspath(p))
                except OSError:
                    pass

    def _cuda_actually_used(self):
        # get_available_providers() only says CUDA was compiled in; a session
        # that failed provider creation silently runs on CPU. Probe the
        # smallest cached model file for the session's real provider list.
        import onnxruntime as ort
        probe = min(self.model_root.glob("**/*.onnx"),
                    key=lambda p: p.stat().st_size, default=None)
        if probe is None:
            return False
        session = ort.InferenceSession(
            str(probe), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return session.get_providers()[0] == "CUDAExecutionProvider"

    def _create(self, providers, quantization=None):
        if "CUDAExecutionProvider" in providers:
            self._preload_gpu_dlls()
        import onnx_asr
        return onnx_asr.load_model(
            self.model_name, path=self.model_root,
            providers=providers, quantization=quantization,
        )

    def _load(self):
        warm = np.zeros(SAMPLE_RATE, dtype=np.float32)
        try:
            self.model = self._create(["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.model.recognize(warm, sample_rate=SAMPLE_RATE)
            if not self._cuda_actually_used():
                raise RuntimeError("CUDA provider fell back to CPU")
            self.provider = "CUDA"
            print(f"[live] Parakeet warm; provider={self.provider}")
            return
        except Exception as e:
            print(f"[live] CUDA Parakeet failed ({e}); falling back to CPU int8")
            self.model = None
        self.model = self._create(["CPUExecutionProvider"], quantization="int8")
        self.model.recognize(warm, sample_rate=SAMPLE_RATE)
        self.provider = "CPU-int8"
        print("[live] WARNING: Parakeet provider=CPU-int8; latency may exceed budget")

    def decode(self, audio):
        started = time.perf_counter()
        text = self.model.recognize(np.asarray(audio, dtype=np.float32),
                                    sample_rate=SAMPLE_RATE)
        elapsed = time.perf_counter() - started
        duration = len(audio) / SAMPLE_RATE
        rtf = elapsed / duration if duration else 0.0
        print(f"[live] decode={elapsed * 1000:.0f}ms rtf={rtf:.3f} provider={self.provider}")
        return str(text).strip(), elapsed


class LiveWorker:
    """Keyboard-free VAD/segmentation/ASR/command worker."""

    def __init__(self, recorder, router, store, cfg, *, engine=None, vad=None,
                 speak_fn=None, response_fn=None, state_fn=None, ignored_fn=None,
                 heard_fn=None, awaiting_fn=None, clock=None):
        self.recorder = recorder
        self.router = router
        self.store = store
        self.cfg = cfg
        self.engine = engine or ParakeetEngine(
            cfg["engine"], Path(__file__).parent / "data" / "models" / "parakeet")
        self.vad = vad or SileroVad()
        self.speak_fn = speak_fn or (lambda _text: None)
        self.response_fn = response_fn or (lambda _text: None)
        self.state_fn = state_fn or (lambda: None)
        self.ignored_fn = ignored_fn or (lambda _text: None)
        self.heard_fn = heard_fn or (lambda _text, _command: None)
        self.awaiting_fn = awaiting_fn or (lambda _active, _deadline: None)
        self.clock = clock or time.monotonic
        self._q = recorder.subscribe(maxsize=96)
        self._lock = threading.Lock()
        self._enabled = False
        self._pauses = set()
        self._generation = 0
        self._reset_pending = True
        self.awaiting_command = False
        self.awaiting_command_deadline = 0.0
        threading.Thread(target=self._loop, daemon=True, name="live-worker").start()

    @property
    def enabled(self):
        with self._lock:
            return self._enabled

    @property
    def listening(self):
        with self._lock:
            return self._enabled and not self._pauses

    def start(self):
        with self._lock:
            if self._enabled:
                return
            self._enabled = True
            self._cancel_awaiting_locked()
            self._invalidate_locked()
        self._drain()
        self.state_fn()
        print("[live] listening")

    def stop(self):
        with self._lock:
            if not self._enabled:
                return
            self._enabled = False
            self._cancel_awaiting_locked()
            self._invalidate_locked()
        self._drain()
        self.awaiting_fn(False, 0.0)
        self.state_fn()
        print("[live] off")

    def pause(self, reason="external"):
        with self._lock:
            self._pauses.add(reason)
            self._cancel_awaiting_locked()
            self._invalidate_locked()
        self._drain()
        self.awaiting_fn(False, 0.0)
        self.state_fn()

    def resume(self, reason="external"):
        with self._lock:
            self._pauses.discard(reason)
            self._invalidate_locked()
        self._drain()
        self.state_fn()

    def close(self):
        self.stop()
        self.recorder.unsubscribe(self._q)

    def _invalidate_locked(self):
        self._generation += 1
        self._reset_pending = True

    def _cancel_awaiting_locked(self):
        self.awaiting_command = False
        self.awaiting_command_deadline = 0.0

    def _valid(self, generation):
        with self._lock:
            return (self._enabled and not self._pauses
                    and generation == self._generation)

    def _drain(self):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _reset(self):
        self.vad.reset()
        self._assembler = np.zeros(0, dtype=np.float32)
        self._pre = deque(maxlen=PRE_FRAMES)
        self._speech = None
        self._silence = 0
        self._speech_end = 0

    def _loop(self):
        self._reset()
        while True:
            item = self._q.get()
            with self._lock:
                active = self._enabled and not self._pauses
                generation = self._generation
                reset = self._reset_pending
                self._reset_pending = False
            if reset or item is None:
                self._reset()
                if item is None:
                    continue
            if not active:
                continue
            chunk = np.asarray(item, dtype=np.float32).reshape(-1)
            self._assembler = np.concatenate((self._assembler, chunk))
            while len(self._assembler) >= FRAME_SAMPLES:
                frame = self._assembler[:FRAME_SAMPLES]
                self._assembler = self._assembler[FRAME_SAMPLES:]
                self._frame(frame, generation)

    def _frame(self, frame, generation):
        speech = self.vad.probability(frame) >= 0.5
        if self._speech is None:
            if speech:
                self._speech = list(self._pre) + [frame]
                self._silence = 0
                self._speech_end = len(self._speech)
            else:
                self._pre.append(frame)
            return
        self._speech.append(frame)
        if speech:
            self._silence = 0
            self._speech_end = len(self._speech)
        else:
            self._silence += 1
        silence_frames = max(1, int(np.ceil(self.cfg["silence_ms"] / 32)))
        hard_cap = len(self._speech) >= int(self.cfg["max_utterance_s"] * SAMPLE_RATE / FRAME_SAMPLES)
        if self._silence >= silence_frames or hard_cap:
            end = min(len(self._speech), self._speech_end + POST_FRAMES)
            audio = np.concatenate(self._speech[:end]).astype(np.float32)
            self._reset()
            self._decode_and_act(audio, generation)

    def _decode_and_act(self, audio, generation):
        text, decode_s = self.engine.decode(audio)
        if not self._valid(generation):
            print("[live] discarded stale decode")
            return
        with self._lock:
            now = self.clock()
            follow_up = (self.awaiting_command
                         and now <= self.awaiting_command_deadline)
            self._cancel_awaiting_locked()
        command = text.strip() if follow_up else wake_command(
            text, self.cfg.get("wake_word"))
        if command is None:
            print(f"[live] ignored: {text}")
            self.ignored_fn(text)
            return
        self.heard_fn(text, command)
        if command == "":
            self.pause("live-sapi")
            try:
                self.speak_fn("Yes?")
            finally:
                self.resume("live-sapi")
            with self._lock:
                if self._enabled and not self._pauses:
                    self.awaiting_command = True
                    self.awaiting_command_deadline = self.clock() + 8.0
                    deadline = self.awaiting_command_deadline
                else:
                    deadline = 0.0
            self.awaiting_fn(bool(deadline), deadline)
            self.response_fn("Yes?")
            return
        if not self._valid(generation):
            return
        response = self.router.handle(command)
        if not self._valid(generation):
            return
        if self.store:
            self.store.add(
                text, source="pc-live", duration_s=len(audio) / SAMPLE_RATE,
                asr_s=decode_s, model=self.cfg["engine"], device=self.engine.provider,
                audio=audio, sample_rate=SAMPLE_RATE,
            )
        print(f"[live] heard: {text} -> {response}")
        if response:
            self.response_fn(response)
            self.pause("live-sapi")
            try:
                self.speak_fn(response)
            finally:
                self.resume("live-sapi")
