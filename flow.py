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
    # Force UTF-8: Windows default is cp1252, so any print containing a smart
    # dash, arrow, or emoji (common in cleaned transcripts and window titles)
    # would raise UnicodeEncodeError and kill that dictation. errors="replace"
    # makes logging unconditionally safe.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
        if sys.stderr:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
else:
    # pythonw.exe has no console — keep diagnostics by logging to file.
    try:
        _logf = open(Path(__file__).parent / "flow.log", "a",
                     buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = sys.stderr = _logf
    except Exception:
        pass

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
# faster_whisper is imported lazily in Flow._load_model — thin clients don't
# have (or need) the CUDA stack installed.

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
    "mode": "full",                # "full" = local GPU; "client" = thin client
    "server_url": "",              # client mode: URL of a full Flow server
    "fallback_server_url": "",     # optional one-shot fallback ASR node
    "client_name": "",             # client mode: device tag (defaults to hostname)
    # Devices whose dictations join YOUR corpus/Hermes context. Anything else
    # is stored private: no audio archive, excluded from the Hermes sweep.
    "trusted_devices": ["android-bubble", "phone", "work-vm"],
    "weather_location": "",
    "sites": {},                   # custom voice names/URLs merged over built-ins
    "integrations": {"hermes": False},
    # Mic selection. None = follow the Windows default input. A string matches
    # a device NAME (case-insensitive substring) because indices churn.
    "input_device": None,
    "mic_silence_guard": True,     # speak up when the mic returns digital silence
    "mic_autoswitch": True,        # ...and move to an input that actually carries signal
    "hotkey": "f8",                # hold to dictate (paste transcript at cursor)
    "command_hotkey": "right alt",    # hold to issue a spoken COMMAND (acts + speaks back)
    "live_hotkey": "f9",
    "live": {"enabled": True, "wake_word": "flow", "conversation_s": 45.0,
             "silence_ms": 280,
             "max_utterance_s": 12, "engine": "nemo-parakeet-tdt-0.6b-v3"},
    "tts": {"engine": "kokoro", "voice": "af_heart", "speed": 1.05},
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
        "client": True,            # thin clients: stream chunks to the server + live-type
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


# A sleeping/absent mic (wireless headset off, privacy switch on) returns
# DIGITAL silence — measured ~3e-5 peak, while a whisper still peaks above
# 5e-3. Anything under this is a dead input, not quiet speech.
SILENCE_PEAK = 0.002
# Loopback/monitor inputs would "pass" a signal probe while capturing your
# speakers instead of your voice.
NOT_A_MIC = ("speaker", "loopback", "stereo mix", "what u hear",
             "sound mapper", "primary sound capture", "output")


def peak_level(audio):
    return float(np.abs(audio).max()) if audio is not None and len(audio) else 0.0


def probe_input(index, sample_rate=16000, seconds=0.4):
    """Peak level of a short capture from one device (0.0 if it can't open)."""
    try:
        audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                       channels=1, dtype="float32", device=index)
        sd.wait()
        return peak_level(audio)
    except Exception:
        return 0.0


def live_input_devices():
    """Unique real microphones, in enumeration order."""
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    found, seen = [], set()
    for i, d in enumerate(devices):
        name = d["name"]
        key = name.lower()
        if d["max_input_channels"] < 1 or key in seen:
            continue
        if any(bad in key for bad in NOT_A_MIC):
            continue
        seen.add(key)
        found.append((i, name))
    return found


class Recorder:
    """Always-open mic stream with a pre-roll ring buffer.

    Opening the stream on keypress loses ~100-300ms — the first syllable.
    Instead the stream runs permanently: while idle, audio goes into a small
    rolling buffer (RAM only, discarded continuously); on start() the last
    `pre_roll_s` seconds are prepended, so the recording begins in the past.
    """

    def __init__(self, sample_rate, pre_roll_s=0.4, device=None):
        from collections import deque
        self.sample_rate = sample_rate
        self.pre_roll = int(pre_roll_s * sample_rate)
        self._chunks = []
        self._ring = deque()
        self._ring_samples = 0
        self._recording = False
        self._stream = None
        self._subscribers = []
        self.device_spec = device
        self.device_index = None
        self.device_name = None
        self.level = 0.0  # live RMS for the overlay waveform
        self._open()

    @staticmethod
    def resolve_device(spec):
        """Map an input_device config value to (index, name).

        None/'' follows the Windows default. Strings match a device name
        (case-insensitive substring) rather than an index, because indices
        renumber whenever devices come and go."""
        try:
            devices = sd.query_devices()
        except Exception as e:
            print(f"[mic] device query failed: {e}")
            return None, None
        inputs = [(i, d) for i, d in enumerate(devices)
                  if d["max_input_channels"] > 0]
        if isinstance(spec, bool):
            spec = None
        if isinstance(spec, int):
            match = next((d for i, d in inputs if i == spec), None)
            if match:
                return spec, match["name"]
            print(f"[mic] input device {spec} not found; using the default")
        elif isinstance(spec, str) and spec.strip():
            needle = spec.strip().lower()
            match = next(((i, d) for i, d in inputs
                          if needle in d["name"].lower()), None)
            if match:
                return match[0], match[1]["name"]
            print(f"[mic] no input matches {spec!r}; using the default")
        try:
            return None, sd.query_devices(kind="input")["name"]
        except Exception:
            return None, None

    def _open(self, _reinit=True):
        self.device_index, self.device_name = self.resolve_device(self.device_spec)
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32",
                callback=self._on_audio, device=self.device_index,
            )
            self._stream.start()
            print(f"[mic] listening on {self.device_name or 'default input'}")
        except Exception as e:
            self._stream = None
            print(f"[mic] open failed: {e}")
            # PaErrorCode -9996 (Invalid device) means PortAudio's device
            # table went stale — a mic was (un)plugged after init, so cached
            # indices no longer line up. Reinitialize PortAudio so it
            # re-enumerates, then retry ONCE with a fresh index.
            if _reinit and "-9996" in str(e):
                try:
                    sd._terminate()
                    sd._initialize()
                    print("[mic] reinitialized audio backend after device change")
                except Exception as re:
                    print(f"[mic] backend reinit failed: {re}")
                    return
                self._open(_reinit=False)

    def retarget(self, spec):
        """Move to another input device, discarding the stale pre-roll."""
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        self.device_spec = spec
        self._ring.clear()
        self._ring_samples = 0
        self._open()
        return self._stream is not None

    def _on_audio(self, indata, *_):
        chunk = indata.copy()
        try:
            # Peak (not RMS) — shows the transient peaks of the voice, which is
            # what reads as a waveform. RMS averages them out to a flat nub.
            self.level = float(np.abs(indata).max())
        except Exception:
            pass
        for q in tuple(self._subscribers):
            try:
                q.put_nowait(chunk)
            except Exception:
                try:
                    q.get_nowait()
                    q.put_nowait(None)  # discontinuity: subscriber must reset
                except Exception:
                    pass
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

    def subscribe(self, maxsize=96):
        import queue
        q = queue.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, target):
        self._subscribers = [q for q in self._subscribers if q is not target]


class StreamWorker:
    """Transcribes rolling chunks WHILE the hotkey is held, committing text that
    is stable across two consecutive passes (LocalAgreement). At release only
    the uncommitted tail needs decoding, so release-to-paste stays ~constant no
    matter how long the dictation ran."""

    def __init__(self, flow, live=False, source=None):
        self.flow = flow
        self.source = source or flow.recorder  # anything with .snapshot()
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
            snap = self.source.snapshot()
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


class ClientStreamer:
    """Thin-client streaming: upload PCM16 chunks to the server while the key
    is held, live-type the committed text the server returns (committed text
    never retracts, so typing it early is safe)."""

    def __init__(self, flow):
        self.flow = flow
        self.sid = None
        self.committed = ""
        self.typed = 0
        self.sent = 0
        self.saved_clip = None
        self._active = False
        self._thread = None

    def _url(self, path):
        return self.flow.cfg["server_url"].rstrip("/") + path

    def _hdr(self):
        name = (self.flow.cfg.get("client_name")
                or os.environ.get("COMPUTERNAME", "client"))
        return {"Authorization": "Bearer " + self.flow.cfg["server"]["token"],
                "X-Device": f"{name}|{foreground_window()}"[:200]}

    def start(self):
        import requests
        try:
            r = requests.post(self._url("/stream/start"), headers=self._hdr(),
                              timeout=4)
            r.raise_for_status()
            self.sid = r.json()["sid"]
        except Exception as e:
            print(f"[client-stream] unavailable ({e}); one-shot fallback")
            return
        try:
            self.saved_clip = pyperclip.paste()
        except Exception:
            pass
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import requests
        sr = self.flow.cfg["sample_rate"]
        while self._active:
            time.sleep(0.35)
            snap = self.flow.recorder.snapshot()
            new = snap[self.sent:]
            if len(new) < sr * 0.25:
                continue
            pcm = (np.clip(new, -1, 1) * 32767).astype("<i2").tobytes()
            try:
                r = requests.post(self._url(f"/stream/{self.sid}/audio"),
                                  data=pcm, headers=self._hdr(), timeout=6)
                r.raise_for_status()
                self.sent = len(snap)
                self.committed = r.json().get("committed", "")
                self._type_delta()
            except Exception as e:
                print(f"[client-stream] chunk failed: {e}")

    def _type_delta(self):
        chunk = self.committed[self.typed:]
        if not chunk:
            return
        try:
            pyperclip.copy(chunk)
            time.sleep(0.02)
            keyboard.send("ctrl+v")  # user is already holding ctrl — paste is safe
            self.typed = len(self.committed)
        except Exception as e:
            print(f"[client-stream] live type failed: {e}")

    def stop(self):
        """on_release calls this like a StreamWorker; the real work is finish()."""
        self._active = False
        if self._thread:
            self._thread.join(2.0)
        return self.committed, 0

    def finish(self):
        """Stop uploading; final decode on the server. Returns (text, typed)."""
        import requests
        self._active = False
        if self._thread:
            self._thread.join(2.0)
        if not self.sid:
            return None, 0  # streaming never started -> caller does one-shot
        # ship any remaining audio, then finish
        try:
            snap = self.flow.recorder.snapshot()
            new = snap[self.sent:]
            if len(new):
                pcm = (np.clip(new, -1, 1) * 32767).astype("<i2").tobytes()
                requests.post(self._url(f"/stream/{self.sid}/audio"), data=pcm,
                              headers=self._hdr(), timeout=6)
            r = requests.post(self._url(f"/stream/{self.sid}/finish"),
                              headers=self._hdr(), timeout=30)
            r.raise_for_status()
            return r.json().get("text", ""), self.typed
        except Exception as e:
            print(f"[client-stream] finish failed: {e}")
            return None, self.typed

    def restore_clipboard(self):
        if self.saved_clip is not None:
            try:
                pyperclip.copy(self.saved_clip)
            except Exception:
                pass


def describe_target(window):
    """Human 'where the text is going' line from a window title. DISPLAY ONLY —
    reflects what's already on the user's screen; never stored or sent."""
    w = (window or "").strip()
    if not w:
        return ""
    parts = re.split(r"\s+[—\-–|]\s+", w)
    if len(parts) >= 2:
        app = parts[-1].strip()
        rest = " · ".join(p.strip() for p in parts[:-1]).strip()
        return f"→ {app} · {rest}"[:60] if rest else f"→ {app}"[:60]
    return f"→ {w}"[:60]


class Overlay:
    """Always-on-top listening pill: a live mic waveform + a status line +
    a 'where the text is going' context line, at the bottom of the active
    window. Tk runs in its own thread; state arrives via a queue. Reads the
    live mic level through level_fn so the wave moves with your voice."""

    N_BARS = 22
    COLORS = {"rec": "#e0524d", "work": "#f0b428", "say": "#3ec97e",
              "live": "#4f8ee8"}

    def __init__(self, level_fn=None):
        import queue
        self._q = queue.Queue()
        self._empty = queue.Empty
        self._level_fn = level_fn or (lambda: 0.0)
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
        from collections import deque
        bg, fg, sub = "#15171c", "#ececf0", "#8a8f98"
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.96)
        except Exception:
            pass
        root.configure(bg=bg)
        from hud import HudView
        hud = HudView(root, tk)

        BW, GAP, WH = 3, 2, 22
        wave_w = self.N_BARS * (BW + GAP)
        top = tk.Frame(root, bg=bg)
        top.pack(fill="x", padx=13, pady=(9, 3))
        canvas = tk.Canvas(top, width=14 + wave_w, height=WH, bg=bg,
                           highlightthickness=0)
        canvas.pack(side="left")
        dot = canvas.create_oval(1, WH // 2 - 5, 11, WH // 2 + 5,
                                 fill=self.COLORS["rec"], outline="")
        bars = [canvas.create_rectangle(0, 0, 0, 0, fill=self.COLORS["rec"],
                                        outline="") for _ in range(self.N_BARS)]
        status = tk.Label(top, text="Listening", fg=fg, bg=bg,
                          font=("Segoe UI", 10))
        status.pack(side="left", padx=(9, 16))
        context = tk.Label(root, text="", fg=sub, bg=bg, font=("Segoe UI", 8),
                           anchor="w", justify="left")
        context.pack(fill="x", padx=15, pady=(0, 8))

        levels = deque([0.0] * self.N_BARS, maxlen=self.N_BARS)
        st = {"mode": "idle"}
        agc = {"ref": 0.08}  # decaying reference peak for auto-gain
        hide_at = [None]
        revert_mode = [None]

        def place():
            root.update_idletasks()
            w, h = root.winfo_reqwidth(), root.winfo_reqheight()
            try:
                import ctypes
                from ctypes import wintypes
                hw = ctypes.windll.user32.GetForegroundWindow()
                r = wintypes.RECT()
                if hw and ctypes.windll.user32.GetWindowRect(hw, ctypes.byref(r)) \
                        and r.right - r.left > 200 and r.bottom - r.top > 150:
                    root.geometry(f"+{(r.left + r.right) // 2 - w // 2}"
                                  f"+{r.bottom - h - 26}")
                    return
            except Exception:
                pass
            root.geometry(f"+{(root.winfo_screenwidth() - w) // 2}"
                          f"+{root.winfo_screenheight() - 120}")

        def draw_wave():
            color = self.COLORS.get(st["mode"], "#e0524d")
            live = st["mode"] == "rec"
            raw = self._level_fn() if live else 0.0
            # Auto-gain: normalize against a decaying recent peak so the wave
            # fills the pill whatever the mic gain is; noise gate keeps silence
            # flat; **0.6 curve lifts quiet speech into visible range.
            agc["ref"] = max(raw, agc["ref"] * 0.93, 0.03)
            norm = 0.0 if raw < 0.015 else min(1.0, (raw / agc["ref"]) ** 0.6)
            levels.append(norm)
            for i, rect in enumerate(bars):
                h = max(2, levels[i] * WH)
                cx = 15 + i * (BW + GAP)
                canvas.coords(rect, cx, WH / 2 - h / 2, cx + BW, WH / 2 + h / 2)
                canvas.itemconfig(rect, fill=color)

        def show(mode):
            st["mode"] = mode
            canvas.itemconfig(dot, fill=self.COLORS.get(mode, "#e0524d"))
            root.deiconify()

        def poll():
            try:
                while True:
                    s = self._q.get_nowait()
                    kind = s[0] if isinstance(s, tuple) else s
                    if kind == "hud":
                        hud.handle(s[1:])
                        continue
                    hide_at[0] = None
                    revert_mode[0] = None
                    status.config(fg=fg)
                    if kind == "rec":
                        status.config(text="Listening")
                        context.config(
                            text=describe_target(s[1] if isinstance(s, tuple)
                                                 and len(s) > 1 else ""))
                        hide_at[0] = None
                        show("rec")
                        place()
                    elif kind == "live":
                        status.config(text="Live listening")
                        context.config(text="Say the wake word, then a command")
                        hide_at[0] = None
                        show("live")
                        place()
                    elif kind == "work":
                        status.config(text="Transcribing…")
                        context.config(text="")
                        hide_at[0] = None
                        show("work")
                    elif kind == "say":
                        status.config(text=s[1][:200])
                        context.config(text="")
                        show("say")
                        place()
                        hide_at[0] = time.time() + max(4, min(12, len(s[1]) / 18))
                    elif kind == "ignored":
                        heard = (s[1] or "")[:40]
                        status.config(text=heard or "Heard something", fg=sub)
                        context.config(text="say 'flow' first", fg=sub)
                        show("live")
                        place()
                        hide_at[0] = time.time() + 1.5
                        revert_mode[0] = "live"
                    else:  # idle
                        if hide_at[0] is None:
                            st["mode"] = "idle"
                            root.withdraw()
            except self._empty:
                pass
            except Exception as e:
                print(f"[overlay] poll error (continuing): {e}")
            if st["mode"] in ("rec", "work", "live"):
                draw_wave()
            hud_expanded = hud.update()
            if hud_expanded and st["mode"] == "live":
                root.withdraw()
            elif (not hud_expanded and hud.state.enabled
                  and st["mode"] == "live"):
                root.deiconify()
                place()
            if hide_at[0] is not None and time.time() > hide_at[0]:
                hide_at[0] = None
                if revert_mode[0] == "live":
                    revert_mode[0] = None
                    status.config(text="Live listening", fg=fg)
                    context.config(text="Say the wake word, then a command", fg=sub)
                    show("live")
                    place()
                else:
                    st["mode"] = "idle"
                    root.withdraw()
            root.after(45, poll)  # ~22fps waveform; always reschedule

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
            import comtypes
            comtypes.CoInitialize()  # mute/restore may run on different threads
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
    # The cloud fallback never sees the window title / app context — only the
    # generic filler-strip prompt. Local Ollama gets the full styled prompt.
    out, lane = llm.chain(c, system, text, cloud_system=CLEANUP_PROMPT,
                          local_only=cfg.get("local_only", True))
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
                                 pre_roll_s=self.cfg.get("pre_roll_s", 0.4),
                                 device=self.cfg.get("input_device"))
        self._last_mic_probe = 0.0
        self.muter = OutputMuter()
        self.muter.ensure_unmuted()
        self.recording = False
        self.active_mode = "dictate"
        self.streamer = None
        self._pressed_keys = set()
        self._active_ptt_key = None
        import queue as _queue
        self._ctrl_q = _queue.Queue()
        self._vm_q = _queue.Queue()  # hands VM-triggered transcripts to the server
        self.busy = threading.Lock()
        self.model_lock = threading.Lock()  # serialize GPU across hotkey + server
        self._remote_gen = 0  # remote session counter (stale-timer guard)
        # Session transitions happen on two threads (hotkey ctrl-loop and HTTP
        # request threads); this lock makes each check-then-mutate atomic.
        self._session_lock = threading.Lock()
        self.router = Router(self.cfg)
        threading.Thread(target=self._ctrl_loop, daemon=True,
                         name="hotkey-ctrl").start()
        self.started_at = time.time()
        self.last_pull = None  # last delta sync by multivac (server sets this)
        self.tray = None
        self.overlay = (Overlay(level_fn=lambda: self.recorder.level)
                        if self.cfg.get("overlay", True) else None)
        self.model = None
        self.live_worker = None
        if self.cfg["mode"] == "client":
            # Thin client: no model, no store (the full server stores), no GPU.
            self.store = None
            if not self.cfg["server_url"]:
                sys.exit("client mode needs server_url in config.json")
            print(f"Client mode -> {self.cfg['server_url']}. "
                  f"Hold [{self.cfg['hotkey']}] to dictate, "
                  f"[{self.cfg.get('command_hotkey')}] for a command.")
            return
        h = self.cfg["history"]
        self.store = TranscriptStore(
            APP_DIR / h["dir"], archive_audio=h["archive_audio"]
        ) if h["enabled"] else None
        import speech
        speech.configure(self.cfg.get("tts"))
        t0 = time.time()
        self.model = self._load_model()
        if self.cfg.get("live", {}).get("enabled", True):
            try:
                from live import LiveWorker
                live_cfg = dict(self.cfg["live"])
                live_cfg["vocab"] = list(self.cfg.get("vocab") or ())
                self.live_worker = LiveWorker(
                    self.recorder, self.router, self.store, live_cfg,
                    speak_fn=speak, response_fn=self._show_live_response,
                    state_fn=self.restore_base_state,
                    ignored_fn=self._show_live_ignored,
                    heard_fn=self._show_live_heard,
                    awaiting_fn=self._show_live_awaiting,
                    conversation_fn=self._show_live_conversation)
                self.router.configure_workflows(
                    progress=self._show_workflow_progress,
                    complete=self.live_worker.deliver_async,
                    ask=self.live_worker.ask_workflow)
                self.router.configure_agents(
                    progress=self._show_agent_progress,
                    complete=self._complete_agent)
            except Exception as e:
                print(f"[live] disabled: model/VAD startup failed ({e})")
        if self.cfg["cleanup"]["enabled"] and self.cfg["cleanup"]["provider"] == "ollama":
            threading.Thread(target=self._warm_ollama, daemon=True).start()
        print(f"Model ready in {time.time() - t0:.1f}s. "
              f"Hold [{self.cfg['hotkey']}] to dictate, "
              f"[{self.cfg.get('command_hotkey')}] for a command; "
              f"[{self.cfg.get('live_hotkey')}] toggles live mode.")

    def _warm_ollama(self):
        import requests
        c = self.cfg["cleanup"]
        while True:
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
                return
            except Exception as e:
                print(f"[cleanup] warmup failed ({e}) — retrying in 60s")
                time.sleep(60)

    def mic_is_deaf(self, audio):
        """True when the captured audio is digital silence.

        A sleeping wireless headset stays the Windows default input and yields
        pure zeros, so every dictation returned an empty transcript and the app
        looked like a dead hotkey for a whole session. Say it out loud instead,
        and move to an input that actually carries signal."""
        if not self.cfg.get("mic_silence_guard", True):
            return False
        if peak_level(audio) >= SILENCE_PEAK:
            return False
        name = self.recorder.device_name or "the microphone"
        print(f"[mic] silent capture from {name}")
        switched = (self.recover_mic() if self.cfg.get("mic_autoswitch", True)
                    else None)
        msg = (f"{name} was silent. Switched to {switched}." if switched else
               f"{name} is silent. Check that it's on and unmuted.")
        if self.overlay:
            self.overlay.set(("say", msg))
        speak(msg)
        return True

    def recover_mic(self, min_gap_s=20.0):
        """Switch to the input device carrying the strongest signal."""
        now = time.time()
        if now - self._last_mic_probe < min_gap_s:
            return None
        self._last_mic_probe = now
        current = (self.recorder.device_name or "").lower()
        scored = []
        for index, name in live_input_devices():
            if name.lower() == current:
                continue
            level = probe_input(index, self.cfg["sample_rate"])
            print(f"[mic] probe {name}: peak {level:.5f}")
            if level >= SILENCE_PEAK:
                scored.append((index, name, level))
        if not scored:
            print("[mic] no live input found")
            return None
        # One physical mic appears once per host API (MME, WASAPI, WDM-KS) with
        # slightly different levels. Among comparable candidates take the
        # LOWEST index: those are MME/DirectSound, the ones that reliably open
        # at our sample rate. Picking the loudest can land on a KS endpoint
        # that opens and then delivers nothing.
        loudest = max(level for _, _, level in scored)
        index, name, level = min(
            (c for c in scored if c[2] >= loudest * 0.7), key=lambda c: c[0])
        if not self.recorder.retarget(index):
            return None
        print(f"[mic] switched to {name} (peak {level:.5f})")
        return name

    def startup_mic_check(self):
        """Log — and quietly fix — a dead default input before first use.

        Runs on a daemon thread with `busy` held: probing/retargeting must not
        interleave with a recording, and every other retarget caller already
        runs under `busy` — on_press and remote_start both bail while it's
        locked, so a press during the (normally 0.3s) probe is dropped rather
        than corrupted."""
        if self.cfg["mode"] == "client" or not self.cfg.get("mic_silence_guard", True):
            return
        with self.busy:
            with self._session_lock:
                if self.recording:
                    return  # a session beat us to the mic; leave it alone
            level = probe_input(self.recorder.device_index,
                                self.cfg["sample_rate"], 0.3)
            name = self.recorder.device_name or "default input"
            if level >= SILENCE_PEAK / 4:  # room tone clears this; silence doesn't
                print(f"[mic] {name} live at startup (peak {level:.5f})")
                return
            print(f"[mic] {name} is silent at startup (peak {level:.5f})")
            if self.cfg.get("mic_autoswitch", True):
                self.recover_mic(min_gap_s=0.0)

    def set_icon(self, state, context=None):
        if self.tray:
            self.tray.icon = make_icon(state)
        if self.overlay:
            # Enrich the listening state with the target window so the pill can
            # show "→ where the text is going" (display-only, never sent).
            if state == "rec":
                self.overlay.set(("rec", context or foreground_window()))
            else:
                self.overlay.set(state)

    def restore_base_state(self):
        if self.recording:
            state = "rec"
        elif self.live_worker and self.live_worker.listening:
            state = "live"
        else:
            state = "idle"
        self.set_icon(state)
        if self.overlay:
            self.overlay.set(("hud", "enabled",
                              bool(self.live_worker and self.live_worker.enabled)))
        if self.tray:
            suffix = f"; live [{self.cfg.get('live_hotkey')}] ON" \
                if self.live_worker and self.live_worker.enabled else ""
            self.tray.title = f"Flow — hold {self.cfg['hotkey']} to dictate{suffix}"

    def _show_live_response(self, response):
        if self.overlay and response:
            self.overlay.set(("hud", "response", response))

    def _show_live_ignored(self, text):
        if self.overlay:
            self.overlay.set(("hud", "ignored", text or ""))

    def _show_live_heard(self, text, command):
        if self.overlay:
            self.overlay.set(("hud", "heard", text or "", command or ""))

    def _show_live_awaiting(self, active, deadline):
        if self.overlay:
            self.overlay.set(("hud", "awaiting", bool(active), deadline))

    def _show_live_conversation(self, active, deadline):
        if self.overlay:
            self.overlay.set(("hud", "conversation", bool(active), deadline))

    def _show_workflow_progress(self, name, index, total, description):
        if self.overlay:
            self.overlay.set(("hud", "workflow", name, index, total,
                              description))

    def _show_agent_progress(self, lane, tail):
        if self.overlay:
            self.overlay.set(("hud", "agent", lane, tail))

    def _complete_agent(self, result):
        from agents import short_summary
        outcome, text = result["outcome"], result.get("text") or ""
        if outcome == "ok":
            spoken = short_summary(text) or f"{result['lane']} finished."
            full = text[:400] or spoken
        elif outcome == "cancelled":
            spoken = f"The {result['lane']} agent was cancelled."
            full = spoken
        else:
            reason = (" ".join(text.split())[:120] or outcome).rstrip(".!?")
            spoken = f"The {result['lane']} agent failed: {reason}."
            full = spoken
        self.live_worker.deliver_async(spoken)
        if self.overlay:
            self.overlay.set(("hud", "agent_result", full))

    def on_press(self, mode="dictate"):
        import speech
        speech.stop()
        with self._session_lock:
            if self.recording or self.busy.locked():
                return
            self.active_mode = mode
            self.recording = True
            if self.live_worker:
                self.live_worker.pause("recorder-owner")
            self.recorder.start()  # first — capture must not wait on mute/icon
            self.set_icon("rec")
            if self.cfg.get("mute_output_while_recording"):
                self.muter.mute()
            if mode == "dictate" and self.cfg["streaming"]["enabled"]:
                if self.cfg["mode"] == "client":
                    if self.cfg["streaming"].get("client", True):
                        self.streamer = ClientStreamer(self)
                        self.streamer.start()
                    return
                # Live typing only where no style profile applies — styled text
                # must wait for the whole utterance.
                live = (self.cfg["streaming"].get("live_type", True)
                        and not (self.cfg["cleanup"]["enabled"]
                                 and match_profile(self.cfg, foreground_window())))
                self.streamer = StreamWorker(self, live=live)
                self.streamer.start()

    def on_release(self):
        with self._session_lock:
            if not self.recording:
                return
            if self.active_mode == "remote":
                return  # remote session owns the recorder; the hotkey must not hijack it
            self.recording = False
            streamer = self.streamer
            self.streamer = None
            if streamer:
                streamer.stop()
            audio = self.recorder.stop()
            self.muter.restore()
            mode = self.active_mode
            if self.live_worker and mode != "command":
                self.live_worker.resume("recorder-owner")
        if mode == "command":
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
            # Unknown remote devices (e.g. a family member's thin client) are
            # PRIVATE by default: transcript stored for their history page, but
            # no audio archive and the Hermes sweep skips it.
            dev = (window or "").split("|")[0].lower()
            private = (source == "remote" and
                       not any(t in dev for t in self.cfg["trusted_devices"]))
            self.store.add(raw, source=source, window=window, duration_s=dur,
                           asr_s=asr_t, model=self.cfg["model"],
                           device=self.cfg["device"],
                           audio=None if private else audio,
                           sample_rate=target_sr, cleaned=text, private=private)
        print(f"[{source}] [{dur:.1f}s audio -> {asr_t:.2f}s asr] {text}")
        return text, {"asr_s": round(asr_t, 2), "duration_s": round(dur, 2)}

    def process(self, audio, streamer=None):
        with self.busy:
            self.set_icon("work")
            try:
                dur = len(audio) / self.cfg["sample_rate"]
                if dur < self.cfg["min_duration_s"]:
                    return
                if self.mic_is_deaf(audio):
                    return
                if self.cfg["mode"] == "client":
                    if isinstance(streamer, ClientStreamer) and streamer.sid:
                        text, typed = streamer.finish()
                        if text is None:  # server lost the session mid-flight
                            text, typed = self.client_transcribe(audio, "/transcribe"), 0
                        remainder = (text or "")[typed:]
                        if remainder:
                            paste_text(remainder, "clipboard",
                                       restore=typed == 0)
                        if typed:
                            time.sleep(0.3)
                            streamer.restore_clipboard()
                    else:
                        text = self.client_transcribe(audio, "/transcribe")
                        if text:
                            paste_text(text, self.cfg["paste_mode"])
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
                self.restore_base_state()

    def process_command(self, audio):
        with self.busy:
            self.set_icon("work")
            try:
                dur = len(audio) / self.cfg["sample_rate"]
                if dur < self.cfg["min_duration_s"]:
                    return
                if self.mic_is_deaf(audio):
                    return
                if self.cfg["mode"] == "client":
                    response = self.client_transcribe(audio, "/command",
                                                      field="response")
                else:
                    text, _ = self.transcribe_and_store(
                        audio, source="pc-command", window=foreground_window())
                    if not text:
                        return
                    print(f"[command] heard: {text}")
                    response = self.router.handle(text)
                    if not isinstance(response, str):
                        # Dialogs (Question objects) only work in live mode,
                        # where the next utterance routes back as the answer.
                        response = ("That needs a follow-up answer — use live "
                                    f"mode ({self.cfg.get('live_hotkey')}) for it.")
                print(f"[command] -> {response}")
                if self.overlay and response:
                    self.overlay.set(("say", response))
                speak(response)
            finally:
                if self.live_worker:
                    self.live_worker.resume("recorder-owner")
                self.restore_base_state()

    def client_transcribe(self, audio, route, field="text"):
        """Thin client: ship audio to the main server; on failure fall back to
        the always-on multivac node (plain transcription, /transcribe only)."""
        import requests

        from store import wav_bytes
        name = self.cfg.get("client_name") or os.environ.get("COMPUTERNAME", "client")
        lanes = [self.cfg["server_url"]]
        if route == "/transcribe" and self.cfg.get("fallback_server_url"):
            lanes.append(self.cfg["fallback_server_url"])
        last_err = None
        for i, base in enumerate(lanes):
            try:
                r = requests.post(
                    base.rstrip("/") + route,
                    data=wav_bytes(audio, self.cfg["sample_rate"]),
                    headers={
                        "Authorization": "Bearer " + self.cfg["server"]["token"],
                        "Content-Type": "audio/wav",
                        "X-Device": f"{name}|{foreground_window()}"[:200],
                    },
                    timeout=60,
                )
                r.raise_for_status()
                out = r.json().get(field, "")
                lane = "primary" if i == 0 else "fallback"
                print(f"[client] {route} via {lane} -> {out[:80]!r}")
                return out
            except Exception as e:
                last_err = e
                print(f"[client] {route} lane {i} failed: {e}")
        if self.overlay:
            self.overlay.set(("say", f"Flow unreachable: {last_err}"))
        return ""

    def remote_start(self):
        """Bridge mode (e.g. the work VM): a remote client starts a host-mic
        recording; the transcript is returned to the caller, never pasted here."""
        with self._session_lock:
            if self.recording or self.busy.locked():
                return False
            self.active_mode = "remote"
            self.recording = True
            if self.live_worker:
                self.live_worker.pause("recorder-owner")
            self._remote_gen += 1
            gen = self._remote_gen
            self.recorder.start()
            self.set_icon("rec")
            if self.cfg.get("mute_output_while_recording"):
                self.muter.mute()
            if self.cfg["streaming"]["enabled"]:
                # Host-side streaming for the VM bridge: commits accumulate
                # while the key is held; the daemon polls /record/partial and
                # types committed deltas live.
                self.remote_streamer = StreamWorker(self, live=False)
                self.remote_streamer.start()
            # Orphan guard: if the client dies mid-recording, don't record forever.
            t = threading.Timer(600, lambda: self.remote_cancel(gen))
            t.daemon = True
            t.start()
            return True

    def remote_stop(self, window=None):
        """Returns the transcript, '' if too short/empty, None if no session."""
        with self._session_lock:
            if not (self.recording and self.active_mode == "remote"):
                return None
            self.recording = False
            streamer = getattr(self, "remote_streamer", None)
            self.remote_streamer = None
            committed = streamer.stop() if streamer else ("", 0)
            audio = self.recorder.stop()
            self.muter.restore()
            if self.live_worker:
                self.live_worker.resume("recorder-owner")
        with self.busy:
            self.set_icon("work")
            try:
                if len(audio) / self.cfg["sample_rate"] < self.cfg["min_duration_s"]:
                    return ""
                partial_served = bool(streamer and streamer.committed_text)
                text, _ = self.transcribe_and_store(
                    audio, source="vm-bridge", window=window,
                    committed=committed, skip_cleanup=partial_served)
                if not text:
                    return ""
                # Replacements would shift the already-typed prefix out of
                # alignment — only apply them when nothing was typed live.
                return text if partial_served else apply_replacements(text, self.cfg)
            finally:
                self.restore_base_state()

    def remote_cancel(self, gen=None):
        with self._session_lock:
            if not (self.recording and self.active_mode == "remote"):
                return False
            if gen is not None and gen != self._remote_gen:
                return False  # a newer session owns the recorder now
            self.recording = False
            streamer = getattr(self, "remote_streamer", None)
            self.remote_streamer = None
            if streamer:
                streamer.stop()
            self.recorder.stop()
            self.muter.restore()
            if self.live_worker:
                self.live_worker.resume("recorder-owner")
            self.restore_base_state()
            print("[remote] recording cancelled")
            return True

    def _on_key(self, event):
        # This runs inside the keyboard listener thread — it MUST be trivial and
        # never throw, or the listener dies and every hotkey stops working until
        # restart. So: match, drop auto-repeat, enqueue. All real work (mute,
        # audio start, transcription) happens in _ctrl_loop.
        name = event.name
        if name == self.cfg.get("live_hotkey"):
            if event.event_type == keyboard.KEY_DOWN and name not in self._pressed_keys:
                self._pressed_keys.add(name)
                self._ctrl_q.put(("toggle", "live"))
            elif event.event_type == keyboard.KEY_UP:
                self._pressed_keys.discard(name)
            return
        if name == self.cfg["hotkey"]:
            mode = "dictate"
        elif name == self.cfg.get("command_hotkey"):
            mode = "command"
        else:
            return
        if event.event_type == keyboard.KEY_DOWN:
            if name in self._pressed_keys:
                return  # ignore held-key auto-repeat
            self._pressed_keys.add(name)
            if self._active_ptt_key is None:
                self._active_ptt_key = name
                self._ctrl_q.put(("down", mode))
        elif event.event_type == keyboard.KEY_UP:
            if name not in self._pressed_keys:
                return
            self._pressed_keys.discard(name)
            if name == self._active_ptt_key:
                self._active_ptt_key = None
                self._ctrl_q.put(("up", mode))

    def _ctrl_loop(self):
        while True:
            action, mode = self._ctrl_q.get()
            try:
                if action == "toggle":
                    if not self.live_worker:
                        print("[live] unavailable")
                    elif self.live_worker.enabled:
                        import speech
                        speech.stop()
                        self.live_worker.stop()
                    else:
                        self.live_worker.start()
                elif action == "down":
                    self.on_press(mode)
                else:
                    self.on_release()
            except Exception as e:
                # Never let a press/release error propagate — recover the state.
                print(f"[hotkey] {action} handler error: {e}")
                self.recording = False

    def maybe_start_server(self):
        s = self.cfg["server"]
        if not s["enabled"] or self.cfg["mode"] == "client":
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
        # and never eat the key from other apps. Registered BEFORE the mic
        # probe: the dead-headset probe path scans devices for seconds, and
        # hotkeys must already be live during it.
        keyboard.hook(self._on_key)
        threading.Thread(target=self.startup_mic_check, daemon=True,
                         name="mic-startup-check").start()
        try:
            import webbrowser

            import pystray
            from pystray import MenuItem
            port = self.cfg["server"]["port"]
            self.tray = pystray.Icon(
                "flow", make_icon("idle"),
                "Flow — hold %s to dictate; %s toggles live" %
                (self.cfg["hotkey"], self.cfg.get("live_hotkey")),
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

    def _downgrade_cuda_if_unavailable(self):
        """Use the portable Whisper CPU lane when CUDA is not usable."""
        if str(self.cfg.get("device", "")).lower() != "cuda":
            return
        try:
            import ctranslate2
            available = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            available = False
        if not available:
            print("[asr] CUDA unavailable; falling back to CPU int8")
            self.cfg["device"] = "cpu"
            self.cfg["compute_type"] = "int8"

    def _load_model(self):
        self._downgrade_cuda_if_unavailable()
        print(f"Loading {self.cfg['model']} on {self.cfg['device']}...")
        from faster_whisper import WhisperModel
        model = WhisperModel(
            self.cfg["model"], device=self.cfg["device"],
            compute_type=self.cfg["compute_type"],
        )
        model.transcribe(np.zeros(self.cfg["sample_rate"], dtype=np.float32))
        return model


def make_icon(state):
    from PIL import Image, ImageDraw
    color = {"idle": (90, 90, 90), "rec": (220, 60, 60),
             "work": (240, 180, 40), "live": (70, 130, 225)}[state]
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


def _setup_cuda_status():
    """Return (available, detail) without importing torch."""
    providers, count = [], 0
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception:
        pass
    try:
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
    except Exception:
        pass
    available = count > 0 or "CUDAExecutionProvider" in providers
    name = ""
    if available:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3)
            name = next((line.strip() for line in result.stdout.splitlines()
                         if line.strip()), "")
        except Exception:
            pass
    return available, name or ("available (name unavailable)"
                               if available else "not available")


def _setup_ollama():
    """Return (reachable, model names) from Ollama's local tags endpoint."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",
                                    timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [m.get("name", "") for m in payload.get("models", [])
                 if isinstance(m, dict) and m.get("name")]
        return True, names
    except Exception:
        return False, []


def setup_machine():
    """Diagnose a machine and create its initial config without app side effects."""
    print("Flow setup")
    print(f"Python: {sys.version.split()[0]}")
    cuda, gpu = _setup_cuda_status()
    print(f"CUDA: {'yes' if cuda else 'no'} ({gpu})")

    scored = []
    print("Input devices:")
    for index, name in live_input_devices():
        level = probe_input(index, DEFAULTS["sample_rate"], 0.4)
        state = "SILENT" if level < SILENCE_PEAK else "live"
        print(f"  [{index:>3}] {state:<6} peak {level:.5f}  {name}")
        if level >= SILENCE_PEAK:
            scored.append((level, name))
    if not scored:
        print("  no live input detected")
    input_name = max(scored)[1] if scored else None

    ollama, models = _setup_ollama()
    print("Ollama: " + (f"reachable ({', '.join(models) or 'no models'})"
                        if ollama else "not reachable"))
    kokoro = (APP_DIR / "data" / "models" / "kokoro" /
              "kokoro-v1.0.onnx").is_file()
    parakeet = any((APP_DIR / "data" / "models" /
                    "parakeet").glob("**/*.onnx"))
    print(f"Kokoro cache: {'ready' if kokoro else 'missing'}")
    print(f"Parakeet cache: {'ready' if parakeet else 'missing'}")

    candidate = json.loads(json.dumps(DEFAULTS))
    candidate["input_device"] = input_name
    candidate["cleanup"].update(
        {"enabled": ollama, "provider": "ollama", "model": "qwen3:8b",
         "base_url": "http://127.0.0.1:11434"})
    candidate["server"]["enabled"] = False
    candidate["tts"]["engine"] = "kokoro"
    if CONFIG_PATH.exists():
        print("\nconfig.json exists; changed nothing. A new config would use:")
        print(f"  input_device={input_name!r}, cleanup.enabled={ollama}, "
              "server.enabled=False, tts.engine='kokoro'")
        summary = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    else:
        CONFIG_PATH.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        print("\nWrote config.json.")
        summary = candidate

    missing = []
    if not scored:
        missing.append("No live microphone -> connect or enable a microphone; "
                       "python flow.py --setup")
    if not ollama:
        missing.append("Ollama not found -> winget install Ollama.Ollama; "
                       "ollama pull qwen3:8b")
    elif not any(name.split(":")[0] == "qwen3" for name in models):
        missing.append("Cleanup model missing -> ollama pull qwen3:8b")
    if not kokoro:
        missing.append("Kokoro cache missing -> .venv\\Scripts\\python.exe "
                       "flow.py (downloads it automatically)")
    if not parakeet:
        missing.append("Parakeet cache missing -> .venv\\Scripts\\python.exe "
                       "flow.py (downloads it when live mode starts)")
    print()
    if missing:
        print("Still needed:")
        for number, item in enumerate(missing, 1):
            print(f"  {number}. {item}")
    else:
        print(f"You're ready: hold {summary.get('hotkey', 'f8')} to dictate, "
              f"press {summary.get('live_hotkey', 'f9')} for live mode.")


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
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_machine()
    elif len(sys.argv) > 1 and sys.argv[1] in ("--history", "-H"):
        cli_history(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif len(sys.argv) > 1 and sys.argv[1] in ("--devices", "-d"):
        cfg = load_config()
        index, name = Recorder.resolve_device(cfg.get("input_device"))
        print(f"in use: {name or 'unknown'}\n")
        for i, dev in live_input_devices():
            level = probe_input(i, cfg["sample_rate"], 0.4)
            state = "SILENT" if level < SILENCE_PEAK else "live  "
            print(f"  [{i:>3}] {state} peak {level:.5f}  {dev}")
        print("\nSet \"input_device\" in config.json to a name substring "
              "(e.g. \"Razer Seiren\") or null to follow the Windows default.")
    elif len(sys.argv) > 1 and sys.argv[1] in ("--stats", "-s"):
        cfg = load_config()
        print(TranscriptStore(APP_DIR / cfg["history"]["dir"]).stats())
    else:
        _lock = acquire_single_instance()
        Flow().run()
