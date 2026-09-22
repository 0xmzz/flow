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
    "server_urls": [],             # extra candidates (LAN ip, tailnet ip, hostname);
                                   # probed in parallel, best one remembered
    "fallback_server_url": "",     # optional one-shot fallback ASR node
    "client_name": "",             # client mode: device tag (defaults to hostname)
    # Devices whose dictations join YOUR corpus/Hermes context. Anything else
    # is stored private: no audio archive, excluded from the Hermes sweep.
    "trusted_devices": ["android-bubble", "phone", "work-vm"],
    "weather_location": "",
    "sites": {},                   # custom voice names/URLs merged over built-ins
    "integrations": {"hermes": False},
    "hermes": {
        "enabled": False,
        "url": "http://100.108.50.120:8642",
        "key": "",
        "hotkey": "right shift",
        "speak": True,
        "timeout_s": 30,
        "default_thread_title": "Flow",
        "thread_rules": [],
        "run_intent_prefixes": ["go build", "kick off", "background task"],
    },
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
        # Ollama only: how long the model stays resident in VRAM after a call,
        # and the context it allocates. -1 pins it forever (~10GB for qwen3:8b
        # at 32k); "10m" lets it unload when idle at the cost of a reload on
        # first use. Cleanup/command prompts are a sentence or two, so 4k ctx
        # is plenty and saves ~4.5GB of KV cache.
        "keep_alive": "10m",
        "num_ctx": 4096,
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
    "max_record_s": 600,           # local orphan cap; release and transcribe, don't discard
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
    "export": {
        "enabled": False,
        "dest": "",
        "mode": "folder",
        "git_remote": "",
        "include": {"history": True, "workflows": True, "intents": True,
                    "audio": False},
        "interval_min": 60,
        "redact": [],
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


# Whisper caption outros. Longest first so "thank you so much…" wins over
# "thank you for watching". Matched only as a trailing phrase.
_OUTRO_PHRASES = (
    "thank you so much for watching",
    "thank you for watching",
    "thanks for watching",
    "i'll see you in the next video",
    "i’ll see you in the next video",
    "see you in the next video",
    "like and subscribe",
    "please subscribe",
    "subscribe to my channel",
)
_OUTRO_TRAIL = re.compile(
    r"[\s,]*(?:\band\b\s+)?(?:"
    + "|".join(re.escape(p) for p in _OUTRO_PHRASES)
    + r")\b[^\w]*$",
    re.IGNORECASE,
)
# Known Whisper credit line. It shows up mid-transcript, not only as an outro.
_AMARA_RE = re.compile(
    r"subtitles by the amara\.org community", re.IGNORECASE)
# A sentence ends at . ! ? only when that mark is followed by whitespace or
# the end of the text. A dot inside 5.5, an IP, or a domain is not a boundary.
_SENT_BOUNDARY = re.compile(r"[.!?](?=\s|$)")
_GAP_CHARS = " \t\r\n"


def _cut_spans(text, spans):
    """Drop [start, end) slices. Every other character stays as it was."""
    merged = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if not merged:
        return text
    parts = []
    i = 0
    for a, b in merged:
        parts.append(text[i:a])
        i = b
    parts.append(text[i:])
    return "".join(parts)


def _sentence_spans(text):
    """(start, end) of each sentence. The gap before a sentence is part of it.

    Deleting a later sentence therefore deletes the whitespace in front of
    it, and the kept characters are a literal slice of `text`.
    """
    spans = []
    start = 0
    for m in _SENT_BOUNDARY.finditer(text):
        if m.end() > start:
            spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def _norm_sentence(sentence):
    folded = sentence.replace("’", "'").lower()
    folded = re.sub(r"[^\w\s']", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _amara_spans(text):
    """Cut the Amara credit wherever it sits, plus one side of the gap around it."""
    spans = []
    for m in _AMARA_RE.finditer(text):
        a, b = m.start(), m.end()
        while b < len(text) and text[b] in ".,!?;:":
            b += 1
        left = a > 0 and text[a - 1] in _GAP_CHARS
        if b < len(text) and text[b] in _GAP_CHARS:
            while b < len(text) and text[b] in _GAP_CHARS:
                b += 1
            # The credit ended the text: drop the gap in front of it too.
            if b == len(text):
                left = True
        if left and b == len(text):
            while a > 0 and text[a - 1] in _GAP_CHARS:
                a -= 1
        spans.append((a, b))
    return spans


def _trailing_outro_spans(text):
    spans = []
    end = len(text)
    while end > 0:
        m = _OUTRO_TRAIL.search(text, 0, end)
        if not m or m.end() != end or m.start() >= end:
            break
        spans.append((m.start(), end))
        end = m.start()
    return spans


def _repeat_spans(text):
    """A sentence said ≥3 times in a row keeps the first copy."""
    parts = _sentence_spans(text)
    if len(parts) < 3:
        return []
    norms = [_norm_sentence(text[a:b]) for a, b in parts]
    spans = []
    i = 0
    while i < len(parts):
        j = i + 1
        while j < len(parts) and norms[j] and norms[j] == norms[i]:
            j += 1
        if j - i >= 3:
            spans.append((parts[i + 1][0], parts[j - 1][1]))
        i = j
    return spans


def _short_run_spans(text):
    """Drop a trailing run of ≥3 identical sentences of ≤3 words each."""
    parts = _sentence_spans(text)
    if len(parts) < 3:
        return []
    norms = [_norm_sentence(text[a:b]) for a, b in parts]
    last = norms[-1]
    if not last or len(last.split()) > 3:
        return []
    i = len(parts) - 1
    while i >= 0 and norms[i] == last:
        i -= 1
    if len(parts) - 1 - i >= 3:
        return [(parts[i + 1][0], parts[-1][1])]
    return []


def strip_hallucinations(text):
    """Drop Whisper caption outros and stuttered sentence loops.

    Applied to the newly decoded tail only. Committed (already typed) text
    is not passed in, so a live prefix on screen is left alone.

    Nothing removed means the same string back, byte for byte. A removal
    cuts that span and does not re-split or re-join the rest.
    """
    if not text:
        return text or ""
    cut = _cut_spans(text, _amara_spans(text))
    cut = _cut_spans(cut, _trailing_outro_spans(cut))
    cut = _cut_spans(cut, _short_run_spans(cut) + _repeat_spans(cut))
    return text if cut == text else cut


def apply_replacements(text, cfg):
    """Whole-word, case-insensitive replacements (e.g. btw -> by the way)."""
    reps = cfg.get("replacements") or {}
    for k, v in reps.items():
        text = re.sub(rf"\b{re.escape(k)}\b", lambda _m: v, text, flags=re.IGNORECASE)
    return text


def resolve_profile(cfg, window):
    """The first profile whose match list hits the window title, or None.

    Callers need the whole profile, not just its style: feedback mode (live vs
    ghost typing) is a property of the skill, and deriving it from "does a
    style exist" is what coupled styling to a live-typing regression."""
    w = (window or "").lower()
    for p in cfg["cleanup"].get("profiles", []):
        if any(m in w for m in p.get("match", [])):
            return p
    return None


def match_profile(cfg, window):
    p = resolve_profile(cfg, window)
    return p.get("style") if p else None


def feedback_mode(cfg, window):
    """"live" (paste words as you speak) or "ghost" (rolling decode, no paste).

    Styled text cannot be live-typed — it is not known until the utterance
    ends — so a matched skill defaults to ghost. A skill whose transform
    preserves word order can opt back in with "feedback": "live". Ghost still
    runs the rolling decode, so release-to-paste stays fast either way."""
    if not cfg["cleanup"]["enabled"]:
        return "live"
    p = resolve_profile(cfg, window)
    if p is None:
        return "live"
    # Default mirrors the OLD predicate exactly, which tested the truthiness of
    # the STYLE, not the existence of the profile: a matched profile with no
    # style is a text no-op (llm_cleanup early-returns on style None under
    # mode "matched"), so it must not cost live typing.
    return p.get("feedback", "ghost" if p.get("style") else "live")


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

    def halt(self):
        """Stop committing. Join happens later, off the ctrl-loop thread."""
        self._active = False

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
        # Same lane the one-shot path picked, or streaming would keep aiming at
        # a configured URL that stopped answering while /transcribe worked.
        # resolve_lane() is TTL-cached, so this is free after the first call.
        base = self.flow.resolve_lane() or self.flow.cfg.get("server_url", "")
        return base.rstrip("/") + path

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

    def halt(self):
        """Stop uploading. finish() joins the thread off the ctrl loop."""
        self._active = False

    def stop(self):
        """on_release calls this like a StreamWorker; the real work is finish()."""
        self.halt()
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


def stop_speech():
    """Interrupt any spoken response. No-op where speech isn't installed.

    Called on every hotkey press, so a thin client missing speech.py must not
    raise here — that would break dictation itself."""
    try:
        import speech
        speech.stop()
    except Exception:
        pass


class _NoHud:
    """Stand-in when hud.py isn't present (thin clients)."""

    class _State:
        enabled = False

    state = _State()

    def handle(self, _event):
        return None

    def update(self):
        return False


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
        try:
            from hud import HudView
            hud = HudView(root, tk)
        except Exception as e:
            # Thin clients ship the dictation core without the HUD module; the
            # overlay pill must still work, so fall back to an inert panel
            # rather than killing this Tk thread.
            print(f"[hud] panel unavailable ({e}); overlay pill only")
            hud = _NoHud()

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
                        label = "Transcribing…"
                        if isinstance(s, tuple) and len(s) > 1 and s[1]:
                            label = str(s[1])[:200]
                        status.config(text=label)
                        context.config(text="")
                        hide_at[0] = None
                        show("work")
                        place()
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


# Hotkeys we can check against the real OS key state. Anything else
# (f8, letters, ...) keeps hook-only behaviour: there is no VK to ask.
_HOTKEY_VK = {
    "right ctrl": 0xA3,
    "right alt": 0xA5,
    "right shift": 0xA1,
    "f9": 0x78,
}
_KEY_POLL_S = 0.15          # ctrl-loop idle poll; two polls confirm a lost KEY_UP
_KEY_UP_POLLS = 2           # one GetAsyncKeyState flicker must not cut a dictation
_REPEAT_MAX_GAP_S = 1.0     # Windows auto-repeat is ~30–50ms; a longer gap is a new press
_MIC_WAIT_S = 1.0           # a press waits this long for a probe, then says so
_PINNED_REPROBES = 3        # silent pinned mic, rechecked before a device sweep
_PINNED_REPROBE_S = 2.0
_STACK_RECHECK_S = 15.0     # all-silent boot: the audio stack may not be up yet


def _speech_enabled(cfg):
    """Kokoro is warmed only when something in this config will speak."""
    live = cfg.get("live") or {}
    hermes = cfg.get("hermes") or {}
    return bool(live.get("enabled") or cfg.get("command_hotkey")
                or hermes.get("enabled"))
_WAIT_HINT_S = 1.5          # after this long, the overlay counts the seconds
_WAIT_BUSY_S = 8.0          # after this long, say the machine is busy
_WAIT_TICK_S = 1.0


def _transcribing_label(elapsed):
    """Overlay line for a decode that has already run past _WAIT_HINT_S."""
    whole = max(1, int(elapsed))
    if elapsed >= _WAIT_BUSY_S:
        return f"Still transcribing ({whole}s) — the PC is busy"
    return f"Transcribing… ({whole}s)"


def _vk_for_hotkey(name):
    if not isinstance(name, str):
        return None
    return _HOTKEY_VK.get(name.lower())


def _os_key_down(vk):
    """True when the OS reports vk held, False when up, None if unavailable.

    Tests monkeypatch this. Non-Windows (no user32) returns None so the
    watchdog degrades to hook events only.
    """
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
    except Exception:
        return None


def _new_mic_lock():
    """The recording lock. A plain Lock, released by whichever thread ends the take.

    remote_start acquires it on the HTTP thread; remote_stop and the orphan
    timer release it on other threads. An RLock can only be released by its
    owner, so that release was swallowed and every later press timed out.
    """
    return threading.Lock()


# OpenClipboard fails when another app holds it for a moment. Ten tries
# over ~300ms covers that blip; the dictation is never dropped on the floor.
_CLIP_RETRY_DELAYS = (0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04)


def _copy_clipboard(text):
    """Copy `text`, retrying on any failure. True on success, never raises."""
    last = None
    for delay in (0.0,) + _CLIP_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            pyperclip.copy(text)
            return True
        except Exception as e:
            last = e
    print(f"[paste] clipboard copy failed after "
          f"{1 + len(_CLIP_RETRY_DELAYS)} tries: {last}")
    return False


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
    if not _copy_clipboard(text):
        # The transcript still has to land at the cursor. Typing is slower
        # but it does not depend on the clipboard.
        print("[paste] clipboard unavailable; typing the dictation instead")
        keyboard.write(text)
        return
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
            if not _copy_clipboard(old):
                print("[paste] could not restore the previous clipboard")
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
        self.last_text = ""
        self.active_mode = "dictate"
        self.hermes_context = None
        self.streamer = None
        self._pressed_keys = set()
        self._active_ptt_key = None
        # Hook thread and the ctrl-loop watchdog both touch these. The lock is
        # held only for the set update — _on_key must stay non-blocking.
        self._key_lock = threading.Lock()
        self._key_seen_at = {}     # name -> monotonic time of the last event
        self._os_up_streak = 0      # consecutive polls the OS reported the key up
        self._record_started_at = None
        import queue as _queue
        self._ctrl_q = _queue.Queue()
        self._job_q = _queue.Queue()  # dictations, processed strictly in press order
        self._vm_q = _queue.Queue()  # hands VM-triggered transcripts to the server
        self.busy = threading.Lock()  # processing only; a press must not look at this
        self._mic_lock = _new_mic_lock()
        self._mic_held = False
        self.model_lock = threading.Lock()  # serialize GPU across hotkey + server
        self._remote_gen = 0  # remote session counter (stale-timer guard)
        # Session transitions happen on two threads (hotkey ctrl-loop and HTTP
        # request threads); this lock makes each check-then-mutate atomic.
        self._session_lock = threading.Lock()
        self.router = Router(self.cfg, export_fn=self._export_voice)
        self.hermes = None
        if self.cfg.get("hermes", {}).get("enabled"):
            from hermes import Mouthpiece
            history_dir = self.cfg.get("history", {}).get("dir", "data")
            self.hermes = Mouthpiece(
                self.cfg["hermes"], APP_DIR / history_dir)
        threading.Thread(target=self._job_loop, daemon=True,
                         name="flow-jobs").start()
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
        self._configure_speech()
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
        if self.cfg.get("export", {}).get("enabled"):
            threading.Thread(target=self._export_loop, daemon=True,
                             name="flow-export").start()
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
                          "keep_alive": c.get("keep_alive", "10m"),
                          "options": {"num_ctx": c.get("num_ctx", 4096)},
                          "messages": [{"role": "user", "content": "hi"}]},
                    timeout=300,
                ).raise_for_status()
                print(f"[cleanup] local model {c['model']} warm in {time.time()-t0:.1f}s")
                return
            except Exception as e:
                print(f"[cleanup] warmup failed ({e}) — retrying in 60s")
                time.sleep(60)

    def _export_voice(self):
        if not self.cfg.get("export", {}).get("enabled"):
            raise RuntimeError("data export is off; enable it in config.json first")
        from export import export_now
        return export_now(self.cfg)

    def _export_loop(self):
        from export import export_now
        event = threading.Event()
        while True:
            try:
                minutes = max(
                    1.0, float(self.cfg.get("export", {}).get(
                        "interval_min", 60)))
                event.wait(minutes * 60)
                summary = export_now(self.cfg)
                print(f"[export] scheduled: {summary['entries']} entries, "
                      f"{summary['changed_days']} changed days")
            except Exception as exc:
                print(f"[export] scheduled run failed: {exc}")

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

    def recover_mic(self, min_gap_s=20.0, when_silent="no live input found"):
        """Switch to the input device carrying the strongest signal.

        Holds _mic_lock for the whole sweep so a retarget cannot land in the
        middle of a recording. Callers must not already hold the lock:
        startup_mic_check releases it before calling this.
        """
        with self._mic_lock:
            return self._recover_mic(min_gap_s, when_silent)

    def _recover_mic(self, min_gap_s=20.0, when_silent="no live input found"):
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
            print(f"[mic] {when_silent}")
            return None
        # One physical mic appears once per host API (MME, WASAPI, WDM-KS) with
        # slightly different levels. Among comparable candidates take the
        # LOWEST index: those are MME/DirectSound, the ones that reliably open
        # at our sample rate. Picking the loudest can land on a KS endpoint
        # that opens and then delivers nothing.
        loudest = max(level for _, _, level in scored)
        index, name, level = min(
            (c for c in scored if c[2] >= loudest * 0.7), key=lambda c: c[0])
        with self._session_lock:
            if self.recording:
                print("[mic] recording in progress; not retargeting")
                return None
        if not self.recorder.retarget(index):
            return None
        print(f"[mic] switched to {name} (peak {level:.5f})")
        return name

    def _configure_speech(self):
        """Warm Kokoro only when a speaking feature is actually on.

        Dictation-only (no live, no command hotkey, no Hermes) still speaks
        the rare mic warning through SAPI on the first speak() call.
        """
        import speech
        speech.configure(self.cfg.get("tts"), start=_speech_enabled(self.cfg))

    def startup_mic_check(self):
        """Log — and quietly fix — a dead default input before first use.

        A silent pinned mic is re-probed a few times before any device sweep:
        at boot every endpoint reads as silence because the audio stack is
        not up yet. The waits happen without `busy` or the mic lock held, so
        a press is not stuck behind them. An all-silent sweep is "audio stack
        not ready", checked once more later, not a dead mic.
        """
        if self.cfg["mode"] == "client" or not self.cfg.get("mic_silence_guard", True):
            return
        self._startup_mic_pass(followup=False)

    def _startup_mic_pass(self, followup):
        attempts = 1 if followup else _PINNED_REPROBES
        index = self.recorder.device_index
        name = self.recorder.device_name or "default input"
        for i in range(attempts):
            if i:
                time.sleep(_PINNED_REPROBE_S)
            with self._mic_lock:
                with self._session_lock:
                    if self.recording:
                        return
                level = probe_input(index, self.cfg["sample_rate"], 0.3)
            if level >= SILENCE_PEAK / 4:  # room tone clears this; silence doesn't
                print(f"[mic] {name} live at startup (peak {level:.5f})")
                return
            print(f"[mic] {name} is silent at startup (peak {level:.5f})")
        if not self.cfg.get("mic_autoswitch", True):
            return
        if self.recover_mic(min_gap_s=0.0, when_silent="audio stack not ready"):
            return
        if followup:
            return
        print(f"[mic] audio stack not ready; checking again in "
              f"{_STACK_RECHECK_S:.0f}s")
        t = threading.Timer(
            _STACK_RECHECK_S, lambda: self._startup_mic_pass(followup=True))
        t.daemon = True
        t.start()

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

    def _release_mic(self):
        if not getattr(self, "_mic_held", False):
            return
        self._mic_held = False
        try:
            self._mic_lock.release()
        except RuntimeError:
            pass

    def _note_mic_wait(self):
        print(f"[hotkey] mic probe still running after {_MIC_WAIT_S:.0f}s")
        if self.overlay:
            self.overlay.set((
                "say",
                "Microphone is still being checked. Release and try again."))

    def _submit_job(self, mode, audio, streamer, hermes_context,
                    released_at=None):
        """Hand a finished take to the single worker. Press order is paste order."""
        if mode == "hermes":
            self._job_q.put(("hermes", (audio, hermes_context)))
        elif mode == "command":
            self._job_q.put(("command", (audio,)))
        else:
            self._job_q.put(("dictate", (audio, streamer, released_at)))

    def _job_loop(self):
        while True:
            job = self._job_q.get()
            if job is None:
                return
            kind, args = job
            try:
                if kind == "dictate":
                    self.process(*args)
                elif kind == "command":
                    self.process_command(*args)
                elif kind == "hermes":
                    self.process_hermes(*args)
            except Exception as e:
                # One failed take must not kill the worker; the next press
                # is already queued behind this one.
                print(f"[hotkey] {kind} job failed: {e}")

    def on_press(self, mode="dictate"):
        stop_speech()
        # A second PTT key during a recording must return immediately. The
        # lock is held for the whole take, so waiting out _MIC_WAIT_S would
        # block the ctrl loop and then claim the microphone is being checked.
        with self._session_lock:
            if self.recording:
                return
        # Wait outside _session_lock: a probe may take seconds, and the
        # previous release still needs that lock. `busy` is processing, not
        # the mic — a new press records while the previous take decodes.
        if not self._mic_lock.acquire(timeout=_MIC_WAIT_S):
            with self._session_lock:
                if self.recording:
                    return
            self._note_mic_wait()
            return
        started = False
        try:
            with self._session_lock:
                if self.recording:
                    return
                context = None
                if mode == "hermes":
                    if not self.hermes:
                        return
                    # LOCAL ONLY on this path. on_press runs on the ctrl-loop
                    # thread while holding _session_lock — a network call here
                    # would freeze every hotkey. The thread name comes from
                    # local rules; the session id is resolved in process_hermes.
                    try:
                        window = foreground_window()
                        context = {"window": window,
                                   "title": self.hermes.predict_title(window)}
                    except Exception as exc:                 # never wedge
                        print(f"[pc-hermes] thread predict failed: {exc}")
                        context = {"window": "", "title": "Hermes"}
                    self.hermes_context = dict(context)
                self.active_mode = mode
                self.recording = True
                self._record_started_at = time.monotonic()
                self._mic_held = True
                started = True
                if self.live_worker:
                    self.live_worker.pause("recorder-owner")
                self.recorder.start()  # first — capture must not wait on mute/icon
                self.set_icon("rec", (f"Hermes · {context['title']}"
                                      if context else None))
                if self.cfg.get("mute_output_while_recording"):
                    self.muter.mute()
                if mode == "dictate" and self.cfg["streaming"]["enabled"]:
                    if self.cfg["mode"] == "client":
                        if self.cfg["streaming"].get("client", True):
                            self.streamer = ClientStreamer(self)
                            self.streamer.start()
                        return
                    # Feedback mode belongs to the skill, not to "does a style
                    # exist" — see feedback_mode(). Defaults reproduce today's
                    # behaviour exactly: matched -> ghost, unmatched -> live.
                    live = (self.cfg["streaming"].get("live_type", True)
                            and feedback_mode(self.cfg, foreground_window()) == "live")
                    self.streamer = StreamWorker(self, live=live)
                    self.streamer.start()
        finally:
            if not started:
                self._mic_lock.release()

    def on_release(self):
        with self._session_lock:
            if not self.recording:
                return
            if self.active_mode == "remote":
                return  # remote session owns the recorder; the hotkey must not hijack it
            self.recording = False
            self._record_started_at = None
            streamer = self.streamer
            self.streamer = None
            # Signal only. stop() joins the stream thread for up to 2s and
            # must not run here — this is the ctrl loop, and the next press
            # is waiting behind it. process() joins before it reads the commit.
            if streamer:
                streamer.halt()
            audio = self.recorder.stop()
            self.muter.restore()
            mode = self.active_mode
            hermes_context = self.hermes_context
            self.hermes_context = None
            if self.live_worker and mode not in ("command", "hermes"):
                self.live_worker.resume("recorder-owner")
        # The audio array is already detached. Free the mic before the
        # (possibly slow) decode so the next press can record immediately.
        self._release_mic()
        self._submit_job(mode, audio, streamer, hermes_context,
                         time.monotonic())

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

    def _store_transcript(self, raw, source, window, dur, asr_t, audio,
                          sample_rate, cleaned, private, e2e_s=None,
                          stages=None):
        if not (self.store and raw):
            return None
        return self.store.add(
            raw, source=source, window=window, duration_s=dur, asr_s=asr_t,
            model=self.cfg["model"], device=self.cfg["device"],
            audio=None if private else audio, sample_rate=sample_rate,
            cleaned=cleaned, private=private, e2e_s=e2e_s, stages=stages)

    def _flush_pending(self, meta, e2e_s=None, stages=None):
        pending = (meta or {}).get("pending")
        if not pending:
            return None
        return self._store_transcript(
            pending["raw"], pending["source"], pending["window"], pending["dur"],
            pending["asr_t"], pending["audio"], pending["sample_rate"],
            pending["cleaned"], pending["private"], e2e_s=e2e_s, stages=stages)

    def _note_e2e(self, released_at, stages, audio_s):
        """e2e is release → paste. Log a breakdown when that exceeds 3s."""
        if released_at is None:
            return None
        e2e = round(max(0.0, time.monotonic() - released_at), 3)
        if e2e > 3.0:
            print(f"[slow] e2e {e2e:.1f}s: "
                  f"queue {stages['queue_wait']:.1f} "
                  f"join {stages['stream_join']:.1f} "
                  f"decode {stages['decode']:.1f} "
                  f"paste {stages['paste']:.1f} "
                  f"({audio_s:.1f}s audio)")
        return e2e

    def transcribe_and_store(self, audio, sample_rate=None, source="pc-hotkey",
                             window=None, committed=("", 0), skip_cleanup=False,
                             persist=True):
        """Shared pipeline: resample -> ASR (tail only if streaming committed a
        prefix) -> cleanup -> store. Returns (text, meta).

        persist=False leaves the row in meta["pending"] so the dictate path
        can paste first and write history after.
        """
        target_sr = self.cfg["sample_rate"]
        if sample_rate and sample_rate != target_sr:
            audio = resample_linear(audio, sample_rate, target_sr)
        dur = len(audio) / target_sr
        if self.cfg.get("normalize_gain"):
            peak = float(np.abs(audio).max()) or 1.0
            audio = (audio / peak * 0.95).astype(np.float32)
        committed_text, committed_samples = committed
        # Unknown remote devices (e.g. a family member's thin client) are
        # PRIVATE by default: transcript stored for their history page, but no
        # audio archive and the Hermes sweep skips it. Decided FIRST, before
        # anything personal touches the audio, because trust gates BOTH
        # personalisation channels: the vocab bias below and the LLM restyle
        # further down. Case-folded on both sides — dev is lowercased, the
        # configured entries are whatever the user typed.
        dev = (window or "").split("|")[0].lower()
        private = (source.startswith("remote") and
                   not any(t.lower() in dev for t in self.cfg["trusted_devices"]))
        t0 = time.time()
        tail = audio[committed_samples:]
        # The vocab list is personal: priming Whisper with "Folionomics",
        # "Zubair Zia", "veNFT" makes it hear those words in someone else's
        # mumble — an active corruption of their transcript, not just a style
        # leak. Untrusted speech is primed with its own committed prefix only.
        prompt = ((committed_text[-200:] or None) if private
                  else self.bias_prompt(committed_text))
        segs = self.decode(tail, prompt=prompt) if len(tail) > 0 else []
        # `raw` keeps the decoder output, hallucinations included. The filter
        # touches only the new tail — committed text is already on screen
        # when live-typing, and process() slices text[live_typed:].
        tail_raw = " ".join(s.text.strip() for s in segs).strip()
        raw = (committed_text + " " + tail_raw).strip()
        text = (committed_text + " " + strip_hallucinations(tail_raw)).strip()
        asr_t = time.time() - t0
        cleanup_s = 0.0
        if (text and self.cfg["cleanup"]["enabled"] and not skip_cleanup
                and not private):
            c0 = time.monotonic()
            text = llm_cleanup(text, self.cfg, window=window)
            cleanup_s = time.monotonic() - c0
        meta = {"asr_s": round(asr_t, 2), "duration_s": round(dur, 2),
                "cleanup_s": round(cleanup_s, 3)}
        if raw:
            meta["pending"] = {
                "raw": raw, "source": source, "window": window, "dur": dur,
                "asr_t": asr_t, "audio": None if private else audio,
                "sample_rate": target_sr, "cleaned": text, "private": private,
            }
        if persist:
            self._store_transcript(
                raw, source, window, dur, asr_t,
                None if private else audio, target_sr, text, private)
            meta.pop("pending", None)
        print(f"[{source}] [{dur:.1f}s audio -> {asr_t:.2f}s asr] {text}")
        return text, meta

    def _remember(self, text):
        """Keep the transcript before any paste attempt, so a failed paste
        can still be copied from the tray."""
        if text:
            self.last_text = text

    def _safe_paste(self, text, mode, restore=True):
        """Paste, but never let a clipboard/typing error kill this thread."""
        if not text:
            return
        try:
            paste_text(text, mode, restore=restore)
        except Exception as e:
            print(f"[paste] failed, dictation kept in memory: {e}")
            if self.overlay:
                self.overlay.set((
                    "say",
                    "Couldn't paste that dictation. "
                    "Tray menu: Copy last dictation."))

    def _copy_last_dictation(self):
        text = getattr(self, "last_text", "") or ""
        if not text:
            print("[paste] no dictation to copy")
            if self.overlay:
                self.overlay.set(("say", "No dictation to copy yet"))
            return
        if _copy_clipboard(text):
            print("[paste] copied last dictation")
        elif self.overlay:
            self.overlay.set(("say", "Couldn't copy the last dictation"))

    def _announce_wait(self, started, stop):
        """Enqueue a ticking 'still transcribing' line. Never touches Tk."""
        next_at = started + _WAIT_HINT_S
        while True:
            now = time.monotonic()
            if now < next_at:
                if stop.wait(next_at - now):
                    return
                continue
            if stop.is_set():
                return
            elapsed = time.monotonic() - started
            # A newer take owns the overlay while the mic is open. Keep the
            # clock running so the line returns when that take ends.
            if self.overlay and not self.recording:
                self.overlay.set(("work", _transcribing_label(elapsed)))
            next_at += _WAIT_TICK_S
            if next_at <= time.monotonic():
                next_at = time.monotonic() + _WAIT_TICK_S

    def process(self, audio, streamer=None, released_at=None):
        stop = threading.Event()
        started = time.monotonic() if released_at is None else released_at
        watcher = threading.Thread(
            target=self._announce_wait, args=(started, stop),
            daemon=True, name="flow-wait")
        watcher.start()
        with self.busy:
            self.set_icon("work")
            try:
                work_started = time.monotonic()
                queue_s = (0.0 if released_at is None
                           else max(0.0, work_started - released_at))
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
                        self._remember(text)
                        remainder = (text or "")[typed:]
                        if remainder:
                            self._safe_paste(remainder, "clipboard",
                                             restore=typed == 0)
                        if typed:
                            time.sleep(0.3)
                            streamer.restore_clipboard()
                    else:
                        text = self.client_transcribe(audio, "/transcribe")
                        self._remember(text)
                        self._safe_paste(text, self.cfg["paste_mode"])
                    return
                join_s = 0.0
                if streamer:
                    # Join only after the rolling pass has been halted, so
                    # committed_text cannot move under the final decode.
                    j0 = time.monotonic()
                    committed = streamer.stop()
                    join_s = time.monotonic() - j0
                    live_typed = streamer.typed_len
                else:
                    committed = ("", 0)
                    live_typed = 0
                text, meta = self.transcribe_and_store(
                    audio, source="pc-hotkey", window=foreground_window(),
                    committed=committed, skip_cleanup=live_typed > 0,
                    persist=False)
                self._remember(text)
                p0 = time.monotonic()
                e2e = None
                stages = None
                flushed = False
                try:
                    # A bad replacements entry must not drop the dictation.
                    # Paste the unreplaced text and still write history.
                    try:
                        if text and live_typed:
                            # Part of the text is already on screen — flush the rest.
                            remainder = apply_replacements(
                                text[live_typed:], self.cfg)
                            self._safe_paste(remainder, "clipboard",
                                             restore=False)
                        elif text:
                            self._safe_paste(apply_replacements(text, self.cfg),
                                             self.cfg["paste_mode"])
                    except Exception as e:
                        print(f"[paste] replacements failed, pasting the "
                              f"transcript unchanged: {e}")
                        if text and live_typed:
                            self._safe_paste(text[live_typed:], "clipboard",
                                             restore=False)
                        elif text:
                            self._safe_paste(text, self.cfg["paste_mode"])
                    paste_s = time.monotonic() - p0
                    stages = {
                        "queue_wait": round(queue_s, 3),
                        "stream_join": round(join_s, 3),
                        "decode": meta.get("asr_s") or 0,
                        "cleanup": meta.get("cleanup_s") or 0,
                        "paste": round(paste_s, 3),
                    }
                    e2e = self._note_e2e(released_at, stages,
                                         meta.get("duration_s") or 0)
                    # Disk waits until the paste has been attempted.
                    self._flush_pending(meta, e2e_s=e2e, stages=stages)
                    flushed = True
                    if self.cfg.get("save_last_audio"):
                        save_wav(APP_DIR / "last.wav", audio,
                                 self.cfg["sample_rate"])
                finally:
                    if not flushed:
                        self._flush_pending(meta, e2e_s=e2e, stages=stages)
            finally:
                stop.set()
                watcher.join(timeout=1.0)
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
                    # NEVER styled: the router matches literal phrases
                    # (commands.py), so a skill that compresses or rewords the
                    # utterance silently breaks every voice command.
                    text, _ = self.transcribe_and_store(
                        audio, source="pc-command", window=foreground_window(),
                        skip_cleanup=True)
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

    def process_hermes(self, audio, pending):
        """Local ASR -> Hermes, deliberately bypassing history and llm.chain.

        Runs on its own thread: the session lookup is a network call, so it
        must never sit on the hotkey thread (see on_press).
        """
        with self.busy:
            self.set_icon("work")
            try:
                dur = len(audio) / self.cfg["sample_rate"]
                if dur < self.cfg["min_duration_s"] or self.mic_is_deaf(audio):
                    return
                if self.cfg["mode"] == "client":
                    response = self.client_transcribe(
                        audio, "/hermes", field="reply")
                else:
                    t0 = time.time()
                    segs = self.decode(audio, prompt=self.bias_prompt())
                    text = " ".join(s.text.strip() for s in segs).strip()
                    print(f"[pc-hermes] [{dur:.1f}s audio -> "
                          f"{time.time() - t0:.2f}s asr]")
                    if not text:
                        return
                    picked = self.hermes.resolve((pending or {}).get("window", ""))
                    if not picked:
                        response = "Hermes is unavailable right now."
                        if self.overlay:
                            self.overlay.set(("say", response))
                        if self.cfg["hermes"].get("speak", True):
                            speak(response)
                        return
                    result = self.hermes.converse(picked, text)
                    response = (result.get("reply")
                                or "Hermes is unavailable right now.")
                    # "switch to X" persists through ThreadStore(current=True),
                    # so predict_title picks it up on the next press; nothing
                    # to stash on self here.
                if self.overlay:
                    self.overlay.set(("say", response))
                if self.cfg["hermes"].get("speak", True):
                    speak(response)
            except Exception:
                response = "Hermes is unavailable right now."
                if self.overlay:
                    self.overlay.set(("say", response))
                if self.cfg["hermes"].get("speak", True):
                    speak(response)
            finally:
                if self.live_worker:
                    self.live_worker.resume("recorder-owner")
                self.restore_base_state()

    def _lane_state_path(self):
        return APP_DIR / "data" / "known_servers.json"

    def _load_lane_state(self):
        try:
            d = json.loads(self._lane_state_path().read_text(encoding="utf-8"))
            return (d.get("last") or None), list(d.get("known") or [])
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None, []

    def _save_lane_state(self, last, known):
        try:
            p = self._lane_state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"last": last, "known": known}, indent=2),
                         encoding="utf-8")
        except OSError:
            pass

    def client_lanes(self):
        """Base URLs to try, best-known first, de-duplicated.

        A thin client has to survive three independent moves it cannot be told
        about: the host's DHCP lease changing, the tailnet expiring, and the
        host's own hostname resolving to interfaces the client cannot reach.
        So the list is last-known-good, then whatever is configured, then
        everything the host has previously advertised via /health."""
        last, known = self._load_lane_state()
        out = []
        for u in (last, self.cfg.get("server_url"),
                  *(self.cfg.get("server_urls") or []),
                  *known, self.cfg.get("fallback_server_url")):
            u = (u or "").strip().rstrip("/")
            if u and u not in out:
                out.append(u)
        return out

    def resolve_lane(self, timeout=2.0, ttl=60.0):
        """First candidate whose /health answers, probed in PARALLEL.

        Serial probing costs timeout x dead-lanes before the first byte of
        audio moves; in parallel the whole decision is one timeout. /health is
        unauthenticated by design, so this is cheap. Learns the host's other
        addresses from the response so the next move needs no config edit."""
        import concurrent.futures
        import urllib.request
        now = time.time()
        cached = getattr(self, "_lane_cache", None)
        if cached and now - cached[1] < ttl:
            return cached[0]
        lanes = self.client_lanes()
        if not lanes:
            return None

        def probe(base):
            with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
                return base, json.loads(r.read(4096).decode("utf-8", "replace"))

        winner = advertised = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(probe, b): b for b in lanes}
            for fut in concurrent.futures.as_completed(futures, timeout=timeout + 3):
                try:
                    base, body = fut.result()
                except Exception:
                    continue
                if body.get("ok"):
                    winner, advertised = base, body.get("addresses") or []
                    break
        if winner is None:
            self._lane_cache = None
            return None
        _, known = self._load_lane_state()
        merged = list(dict.fromkeys([*known, *(a.rstrip("/") for a in advertised)]))
        self._save_lane_state(winner, merged)
        self._lane_cache = (winner, now)
        print(f"[client] server lane: {winner}"
              + (f" (+{len(merged)} known)" if merged else ""))
        return winner

    def client_transcribe(self, audio, route, field="text"):
        """Thin client: ship audio to the main server; on failure fall back to
        the always-on multivac node (plain transcription, /transcribe only)."""
        import requests

        from store import wav_bytes
        name = self.cfg.get("client_name") or os.environ.get("COMPUTERNAME", "client")
        # Probe first: betting a whole audio upload on a guess costs a connect
        # timeout per wrong lane, and a dead host is the common case.
        lanes = self.client_lanes()
        best = self.resolve_lane()
        if best:
            lanes = [best] + [u for u in lanes if u != best]
        if route != "/transcribe":
            fb = (self.cfg.get("fallback_server_url") or "").rstrip("/")
            lanes = [u for u in lanes if u != fb]  # node is /transcribe-only
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
                    # (connect, read). An unreachable host must fail in
                    # seconds, not a minute: with a dead primary AND a dead
                    # fallback a single 60s connect timeout per lane left the
                    # client sitting on "transcribing" for two full minutes
                    # with no explanation. Reads still get the full budget —
                    # a long dictation legitimately takes a while to decode.
                    timeout=(5, 60),
                )
                r.raise_for_status()
                out = r.json().get(field, "")
                if i:  # a later lane won — remember it so the next one is direct
                    _, known = self._load_lane_state()
                    self._save_lane_state(base.rstrip("/"), known)
                    self._lane_cache = (base.rstrip("/"), time.time())
                print(f"[client] {route} via {base} -> {out[:80]!r}")
                return out
            except Exception as e:
                last_err = e
                print(f"[client] {route} lane {base} failed: {e}")
        self._lane_cache = None  # stop trusting a lane that just failed
        msg = (f"Flow server unreachable. Tried {len(lanes)}: "
               + ", ".join(lanes[:4]) + (" ..." if len(lanes) > 4 else ""))
        print(f"[client] {msg} last error: {last_err}")
        if self.overlay:
            self.overlay.set(("say", msg))
        return ""

    def remote_start(self):
        """Bridge mode (e.g. the work VM): a remote client starts a host-mic
        recording; the transcript is returned to the caller, never pasted here.

        Still refuses while a local recording is active or a take is decoding
        (`busy`). A probe holding the mic lock also refuses — non-blocking,
        so the HTTP handler never waits out a device sweep.
        """
        if not self._mic_lock.acquire(blocking=False):
            return False
        started = False
        try:
            with self._session_lock:
                if self.recording or self.busy.locked():
                    return False
                self.active_mode = "remote"
                self.recording = True
                self._mic_held = True
                started = True
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
        finally:
            if not started:
                self._mic_lock.release()

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
        self._release_mic()
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
        self._release_mic()
        print("[remote] recording cancelled")
        return True

    def _ptt_mode(self, name):
        if name == self.cfg.get("hotkey"):
            return "dictate"
        if name and name == self.cfg.get("command_hotkey"):
            return "command"
        hermes = self.cfg.get("hermes") or {}
        if hermes.get("enabled") and name == hermes.get("hotkey"):
            return "hermes"
        return None

    def _on_live_key(self, event, name):
        enqueue = False
        with self._key_lock:
            if (event.event_type == keyboard.KEY_DOWN
                    and name not in self._pressed_keys):
                self._pressed_keys.add(name)
                self._key_seen_at[name] = time.monotonic()
                enqueue = True
            elif event.event_type == keyboard.KEY_UP:
                self._pressed_keys.discard(name)
                self._key_seen_at[name] = time.monotonic()
        if enqueue:
            self._ctrl_q.put(("toggle", "live"))

    def _on_ptt_down(self, name, mode):
        now = time.monotonic()
        missed_up = False
        start = False
        gap = 0.0
        with self._key_lock:
            last = self._key_seen_at.get(name)
            self._key_seen_at[name] = now
            if name in self._pressed_keys:
                # Auto-repeat lands every ~30–50ms. A down after a silent
                # second is a new press whose KEY_UP never arrived.
                if last is not None and (now - last) <= _REPEAT_MAX_GAP_S:
                    return
                gap = now - last if last is not None else _REPEAT_MAX_GAP_S
                self._pressed_keys.discard(name)
                if name == self._active_ptt_key:
                    self._active_ptt_key = None
                    self._os_up_streak = 0
                    missed_up = True
            self._pressed_keys.add(name)
            if self._active_ptt_key is None:
                self._active_ptt_key = name
                self._os_up_streak = 0
                start = True
        if missed_up:
            print(f"[hotkey] {name!r} down {gap:.1f}s after the last event; "
                  "treating it as a missed release")
            self._ctrl_q.put(("up", mode))
        if start:
            self._ctrl_q.put(("down", mode))

    def _on_ptt_up(self, name, mode):
        release = False
        with self._key_lock:
            self._key_seen_at[name] = time.monotonic()
            if name not in self._pressed_keys:
                return
            self._pressed_keys.discard(name)
            if name == self._active_ptt_key:
                self._active_ptt_key = None
                self._os_up_streak = 0
                release = True
        if release:
            self._ctrl_q.put(("up", mode))

    def _on_key(self, event):
        # This runs inside the keyboard listener thread — it MUST be trivial and
        # never throw, or the listener dies and every hotkey stops working until
        # restart. So: match, drop auto-repeat, enqueue. All real work (mute,
        # audio start, transcription) happens in _ctrl_loop. The lock below is
        # uncontended and held only for the set update.
        try:
            name = event.name
            if name == self.cfg.get("live_hotkey"):
                self._on_live_key(event, name)
                return
            mode = self._ptt_mode(name)
            if mode is None:
                return
            if event.event_type == keyboard.KEY_DOWN:
                self._on_ptt_down(name, mode)
            elif event.event_type == keyboard.KEY_UP:
                self._on_ptt_up(name, mode)
        except Exception as e:
            print(f"[hotkey] listener error: {e}")

    def _drop_held_key(self):
        """Forget the PTT key we think is held. Caller runs the release path."""
        with self._key_lock:
            name = self._active_ptt_key
            if name:
                self._pressed_keys.discard(name)
                self._key_seen_at.pop(name, None)
            self._active_ptt_key = None
            self._os_up_streak = 0
            return name

    def _poll_key_watchdog(self):
        """Synthesize KEY_UP when the OS says the held PTT key is up, twice.

        A release over an elevated window, the lock screen, or a hook stall
        never delivers KEY_UP. The next real press is then swallowed by the
        auto-repeat guard and the mic stays open. One poll is not enough.
        """
        with self._key_lock:
            name = self._active_ptt_key
            if not name:
                self._os_up_streak = 0
                return
            vk = _vk_for_hotkey(name)
            if vk is None:
                return  # no VK: degrade to today's hook-only behaviour
            down = _os_key_down(vk)
            if down is None or down:
                self._os_up_streak = 0
                return
            self._os_up_streak += 1
            if self._os_up_streak < _KEY_UP_POLLS:
                return
            self._pressed_keys.discard(name)
            self._key_seen_at.pop(name, None)
            self._active_ptt_key = None
            self._os_up_streak = 0
        print(f"[hotkey] {name!r} still marked held but the OS reports it up; "
              "releasing")
        self.on_release()

    def _poll_record_cap(self):
        """Force-release a local recording that has run past max_record_s.

        Remote sessions keep their own cancel timer, and that one discards.
        This path goes through on_release so the audio is transcribed.
        Stopping the recording is what bounds Recorder._chunks.
        """
        if not self.recording or getattr(self, "active_mode", None) == "remote":
            return
        try:
            cap = float(self.cfg.get("max_record_s", 600))
        except (TypeError, ValueError, AttributeError):
            return
        if cap <= 0:
            return
        started = self._record_started_at
        if started is None or (time.monotonic() - started) < cap:
            return
        print(f"[hotkey] local recording exceeded {cap:.0f}s; "
              "releasing for transcription")
        self._drop_held_key()
        self.on_release()

    def _ctrl_loop(self):
        import queue
        while not getattr(self, "_ctrl_stop", False):
            try:
                action, mode = self._ctrl_q.get(timeout=_KEY_POLL_S)
            except queue.Empty:
                if getattr(self, "_ctrl_stop", False):
                    return
                try:
                    self._poll_key_watchdog()
                    self._poll_record_cap()
                except Exception as e:
                    # Don't clear `recording` here: a failed poll must not
                    # orphan the open mic. The next poll retries.
                    print(f"[hotkey] watchdog error: {e}")
                continue
            if getattr(self, "_ctrl_stop", False):
                return
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
                self._record_started_at = None
                self._release_mic()

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

    def _tray_menu_items(self):
        """(label, callback) pairs. Callbacks take no arguments."""
        import webbrowser
        port = self.cfg["server"]["port"]
        return (
            ("Open history page", lambda: webbrowser.open(
                f"http://127.0.0.1:{port}/#tok=" + self.cfg["server"]["token"])),
            ("Copy last dictation", lambda: self._copy_last_dictation()),
            ("Edit config (restart to apply)",
             lambda: os.startfile(str(CONFIG_PATH))),
            ("Open data folder", lambda: os.startfile(str(APP_DIR / "data"))),
            ("Quit", lambda: self.tray.stop()),
        )

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
            import pystray
            from pystray import MenuItem
            self.tray = pystray.Icon(
                "flow", make_icon("idle"),
                "Flow — hold %s to dictate; %s toggles live" %
                (self.cfg["hotkey"], self.cfg.get("live_hotkey")),
                menu=pystray.Menu(*(MenuItem(label, action)
                                    for label, action in self._tray_menu_items())),
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


def cli_export():
    from export import export_now
    summary = export_now(load_config())
    print(json.dumps(summary, ensure_ascii=False))


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
    print("Data export: available, off by default (set export.dest to opt in)")

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
    elif len(sys.argv) > 1 and sys.argv[1] == "--export":
        cli_export()
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
