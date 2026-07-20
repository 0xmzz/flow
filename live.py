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

from workflows import Question
from commands import KNOWN_SITES, ROUTER_VERBS


SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512                 # 32 ms, required by Silero
PRE_FRAMES = 5                      # 160 ms
POST_FRAMES = 5                     # 160 ms
GREETINGS = {"hey", "hi", "ok", "okay", "yo", "so"}
FILLERS = {"yeah", "yes", "ok", "okay", "hmm", "uh", "um", "right", "cool"}
QUIET_PHRASES = {"stop listening", "go to sleep", "that's all", "thats all",
                 "that s all"}


def correct_live_text(text, wake_word="flow", vocab=(), workflow_words=()):
    """Lightly repair Parakeet tokens using only local command vocabulary."""
    raw = str(text or "")
    candidates = []
    for value in (list(ROUTER_VERBS) + list(vocab or ())
                  + list(workflow_words or ()) + list(KNOWN_SITES)):
        candidates.extend(re.findall(r"[\w']+", str(value).lower()))
    candidates = sorted(set(candidates), key=lambda value: (-len(value), value))
    wake = re.sub(r"[^\w]", "", str(wake_word or "").lower())

    index = 0
    def replace(match):
        nonlocal index
        token = match.group(0)
        lowered = token.lower()
        first = index == 0
        index += 1
        if len(lowered) < 3:
            return token
        if first and wake:
            ratio = difflib.SequenceMatcher(None, lowered, wake).ratio()
            if ratio >= 0.7 or lowered in {"floats", "flows", "floor", "flo"}:
                return wake.capitalize() if token[:1].isupper() else wake
        if lowered in candidates:
            return token
        scored = []
        for value in candidates:
            forms = (value, value[:-1]) if value.endswith("s") else (value,)
            score = max(difflib.SequenceMatcher(None, lowered, form).ratio()
                        for form in forms)
            scored.append((score, value))
        if not scored:
            return token
        best = max(score for score, _value in scored)
        matches = [value for score, value in scored if score == best]
        if best < 0.8 or len(matches) != 1:
            return token
        value = matches[0]
        return value.capitalize() if token[:1].isupper() else value
    return re.sub(r"[\w']+", replace, raw)


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
                 heard_fn=None, awaiting_fn=None, conversation_fn=None,
                 clock=None):
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
        self.conversation_fn = conversation_fn or (lambda _active, _deadline: None)
        self.clock = clock or time.monotonic
        self._q = recorder.subscribe(maxsize=96)
        self._lock = threading.Lock()
        self._enabled = False
        self._pauses = set()
        self._generation = 0
        self._reset_pending = True
        self.awaiting_command = False
        self.awaiting_command_deadline = 0.0
        self.awaiting_input = None
        self.awaiting_input_deadline = 0.0
        self._dialog_token = 0
        self.conversation_deadline = 0.0
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
            self._clear_conversation_locked()
            self._invalidate_locked()
        self._reset_router_conversation()
        self._drain()
        self.state_fn()
        print("[live] listening")

    def stop(self):
        with self._lock:
            if not self._enabled:
                return
            self._enabled = False
            self._cancel_awaiting_locked()
            self._clear_conversation_locked()
            self._invalidate_locked()
        self._reset_router_conversation()
        self._drain()
        self.awaiting_fn(False, 0.0)
        self.conversation_fn(False, 0.0)
        self.state_fn()
        print("[live] off")

    def pause(self, reason="external"):
        with self._lock:
            self._pauses.add(reason)
            if reason != "live-sapi":
                self._cancel_awaiting_locked()
                self._clear_conversation_locked()
            self._invalidate_locked()
        if reason != "live-sapi":
            self._reset_router_conversation()
            runner = getattr(self.router, "workflow_runner", None)
            if runner and runner.running:
                runner.cancel()
            agent = getattr(self.router, "agent_runner", None)
            if agent and agent.running:
                agent.cancel()
        self._drain()
        if reason != "live-sapi":
            self.awaiting_fn(False, 0.0)
            self.conversation_fn(False, 0.0)
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
        question = self.awaiting_input
        self.awaiting_command = False
        self.awaiting_command_deadline = 0.0
        self.awaiting_input = None
        self.awaiting_input_deadline = 0.0
        self._dialog_token += 1
        if question and question.cancel:
            try:
                question.cancel()
            except Exception:
                pass

    def _clear_conversation_locked(self):
        self.conversation_deadline = 0.0

    def _reset_router_conversation(self):
        reset = getattr(self.router, "reset_conversation", None)
        if reset:
            reset()

    def _conversation_active_locked(self, now):
        expired = False
        if self.conversation_deadline and now >= self.conversation_deadline:
            self._clear_conversation_locked()
            expired = True
        return bool(self.conversation_deadline), expired

    def _touch_conversation(self):
        seconds = float(self.cfg.get("conversation_s", 45.0) or 0.0)
        with self._lock:
            if seconds <= 0 or not self._enabled or self._pauses:
                return 0.0
            self.conversation_deadline = self.clock() + seconds
            deadline = self.conversation_deadline
        self.conversation_fn(True, deadline)
        return deadline

    def _workflow_words(self):
        store = getattr(self.router, "workflows", None)
        if not store:
            return []
        words = []
        try:
            for recipe in store.list():
                words.append(recipe.get("name", ""))
                words.extend(recipe.get("triggers", ()))
        except Exception:
            pass
        return words

    def _install_question(self, question, timeout_s=30.0):
        with self._lock:
            if not self._enabled or self._pauses:
                if question.cancel:
                    question.cancel()
                return False
            self._cancel_awaiting_locked()
            self.awaiting_input = question
            self.awaiting_input_deadline = self.clock() + timeout_s
            self.awaiting_command = True  # compatibility alias
            self.awaiting_command_deadline = self.awaiting_input_deadline
            self._dialog_token += 1
            token = self._dialog_token
            deadline = self.awaiting_input_deadline
        self.awaiting_fn(True, deadline)

        def expire():
            remaining = max(0.0, deadline - self.clock())
            if remaining:
                time.sleep(remaining)
            with self._lock:
                if token != self._dialog_token or self.clock() < deadline:
                    return
                self._cancel_awaiting_locked()
            self.awaiting_fn(False, 0.0)
        threading.Thread(target=expire, daemon=True,
                         name="live-dialog-timeout").start()
        return True

    def ask_workflow(self, text, cancel_event):
        """Publish a yes/no Question and block a workflow worker safely."""
        answered = threading.Event()
        result = {"value": None}

        def cancel():
            answered.set()

        def handle(answer):
            heard = str(answer).strip().lower().rstrip(".!?")
            if heard in {"yes", "yeah", "yep", "sure"}:
                result["value"] = True
            elif heard in {"no", "nope"}:
                result["value"] = False
            else:
                return Question("Please say yes or no.", handle, cancel)
            answered.set()
            return "Continuing workflow."

        question = Question(text, handle, cancel)
        self.response_fn(text)
        self.pause("live-sapi")
        try:
            self.speak_fn(text)
        finally:
            self.resume("live-sapi")
        if not self._install_question(question, 30.0):
            return None
        while not answered.wait(0.1):
            if cancel_event.is_set():
                with self._lock:
                    if self.awaiting_input is question:
                        self._cancel_awaiting_locked()
                self.awaiting_fn(False, 0.0)
                return None
        return result["value"]

    def deliver_async(self, response):
        """Show and speak a result produced by a background workflow."""
        with self._lock:
            can_speak = self._enabled and not self._pauses
        if can_speak:
            self._present_result(response)
        elif response:
            self.response_fn(response)

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
        text = correct_live_text(
            text, self.cfg.get("wake_word"), self.cfg.get("vocab", ()),
            self._workflow_words())
        wake = wake_command(text, self.cfg.get("wake_word"))
        with self._lock:
            now = self.clock()
            conversation, conversation_expired = self._conversation_active_locked(now)
            question = (self.awaiting_input
                        if now <= self.awaiting_input_deadline else None)
            legacy_follow_up = (self.awaiting_command and question is None
                                and now <= self.awaiting_command_deadline)
            runner = getattr(self.router, "workflow_runner", None)
            agent = getattr(self.router, "agent_runner", None)
            cancel_word = str(text).strip().lower().rstrip(".!?") in {"stop", "cancel"}
            bare_agent_cancel = cancel_word and agent and agent.running
            bare_cancel = cancel_word and runner and runner.running
            if not question and not legacy_follow_up and self.awaiting_command:
                self._cancel_awaiting_locked()
        if conversation_expired:
            self._reset_router_conversation()
        if conversation:
            self._touch_conversation()
        if wake is not None:
            self._touch_conversation()
        normalized = re.sub(r"[^\w\s']", "", wake if wake is not None else text,
                            flags=re.UNICODE).lower().replace("’", "'").strip()
        if (wake is not None or conversation) and normalized in QUIET_PHRASES:
            with self._lock:
                self._cancel_awaiting_locked()
                self._clear_conversation_locked()
            self._reset_router_conversation()
            self.awaiting_fn(False, 0.0)
            self.conversation_fn(False, 0.0)
            self.response_fn("Going quiet.")
            self.pause("live-sapi")
            try:
                self.speak_fn("Going quiet.")
            finally:
                self.resume("live-sapi")
            return
        # Wake-worded commands always win. A bare stop/cancel also reaches an
        # active workflow so long waits can be aborted hands-free.
        if wake is not None:
            command = wake
        elif bare_agent_cancel:
            agent.cancel()
            self.response_fn("Cancelling agent.")
            return
        elif bare_cancel:
            runner.cancel()
            self.response_fn("Cancelling workflow.")
            return
        elif question is not None:
            self._touch_conversation()
            answer = text.strip()
            if answer.lower().rstrip(".!?") in {"cancel", "never mind", "nevermind"}:
                with self._lock:
                    self._cancel_awaiting_locked()
                self.awaiting_fn(False, 0.0)
                self.response_fn(question.cancel_text)
                return
            with self._lock:
                if self.awaiting_input is question:
                    self.awaiting_input = None
                    self.awaiting_input_deadline = 0.0
                    self.awaiting_command = False
                    self.awaiting_command_deadline = 0.0
                    self._dialog_token += 1
            self.awaiting_fn(False, 0.0)
            try:
                result = question.handler(answer)
            except Exception as exc:
                result = f"That dialog failed: {exc}"
            remember = getattr(self.router, "remember_exchange", None)
            if remember:
                remember(answer, result)
            self._present_result(result)
            return
        elif legacy_follow_up:
            with self._lock:
                self._cancel_awaiting_locked()
            self.awaiting_fn(False, 0.0)
            command = text.strip()
        elif conversation:
            command = text.strip()
        else:
            command = None
        if command is None:
            print(f"[live] ignored: {text}")
            self.ignored_fn(text)
            return
        if conversation and wake is None and not command:
            return
        if conversation and wake is None and command.lower().rstrip(".!?") in FILLERS:
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
        handle = getattr(self.router, "handle_live", self.router.handle)
        response = handle(command)
        if not self._valid(generation):
            return
        if self.store:
            self.store.add(
                text, source="pc-live", duration_s=len(audio) / SAMPLE_RATE,
                asr_s=decode_s, model=self.cfg["engine"], device=self.engine.provider,
                audio=audio, sample_rate=SAMPLE_RATE,
            )
        print(f"[live] heard: {text} -> {response}")
        self._present_result(response)

    def _present_result(self, response):
        if isinstance(response, Question):
            self.response_fn(response.text)
            self.pause("live-sapi")
            try:
                self.speak_fn(response.text)
            finally:
                self.resume("live-sapi")
            self._install_question(response, 30.0)
        elif response:
            self.response_fn(response)
            self.pause("live-sapi")
            try:
                self.speak_fn(response)
            finally:
                self.resume("live-sapi")
