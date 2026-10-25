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
import requests
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
    "max_latched_s": 7200,         # hands-free latch cap (2h); a held key still uses max_record_s
    "pre_roll_s": 0.4,             # mic ring buffer prepended on press
    # Bool still accepted (older config.json). hints is the third pill line.
    "overlay": {"enabled": True, "hints": True},
    "voice_edits": {
        "scratch": ["scratch that"],
        # Whole-take only. A longer sentence that contains one is ordinary text.
        "paste_last": [
            "paste my last message",
            "paste last message",
            "paste that again",
            "paste my last dictation",
        ],
    },
    # Final paste follows the window that was focused at press.
    "sticky_target": True,
    # Selection rewrite (alt during a take). Off: alt does not capture or
    # touch the clipboard. The path stays; tests set enabled true.
    "rewrite": {"enabled": False},
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


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _word_loop_spans(text, min_reps=4, max_n=3):
    """A 1-3 word unit repeated >=4 times in a row ("safe, safe, safe, safe")
    keeps the first copy. Real speech rarely says the same word four times
    running; Whisper loops do it constantly on weak audio."""
    toks = [(m.start(), m.end(), m.group().lower()) for m in _WORD_RE.finditer(text)]
    spans = []
    i = 0
    while i < len(toks):
        best = None
        for n in range(1, max_n + 1):
            unit = [t[2] for t in toks[i:i + n]]
            if len(unit) < n:
                break
            reps = 1
            while [t[2] for t in toks[i + reps * n:i + (reps + 1) * n]] == unit:
                reps += 1
            if reps >= min_reps and (best is None or reps * n > best[0] * best[1]):
                best = (reps, n)
        if best:
            reps, n = best
            end_tok = toks[i + reps * n - 1]
            # keep the first copy up to (not including) its trailing separator
            spans.append((toks[i + n - 1][1], end_tok[1]))
            i += reps * n
        else:
            i += 1
    return spans


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
    cut = _cut_spans(cut, _word_loop_spans(cut))
    return text if cut == text else cut


def apply_replacements(text, cfg):
    """Whole-word, case-insensitive replacements (e.g. btw -> by the way)."""
    reps = cfg.get("replacements") or {}
    for k, v in reps.items():
        text = re.sub(rf"\b{re.escape(k)}\b", lambda _m: v, text, flags=re.IGNORECASE)
    return text


def _scratch_phrases(cfg):
    if isinstance(cfg, dict):
        phrases = (cfg.get("voice_edits") or {}).get("scratch")
        if phrases:
            return [p for p in phrases if p]
    return list(DEFAULTS["voice_edits"]["scratch"])


def _scratch_pattern(phrases):
    body = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    # Trailing punctuation is part of the command, not the next sentence.
    return re.compile(rf"\b(?:{body})\b[.!?…,;:]*", re.IGNORECASE)


def _scratch_cut(text, match_start, match_end):
    """Where a scratch command deletes, measured from the command's position.

    The command does not have to be its own sentence. When it begins a
    sentence ("Drop. Scratch that and continue."), the sentence that ends
    just before it is the one that goes, and the words after the command
    stay. When it sits mid-sentence, the words before it in that sentence
    are what was just said. A command with no preceding sentence removes
    only itself.
    """
    spans = _sentence_spans(text)
    idx = None
    for i, (s, e) in enumerate(spans):
        if s <= match_start < e or (match_start == e and e == len(text)):
            idx = i
            break
    if idx is None:
        return match_start, match_end
    s, _e = spans[idx]
    at_start = text[s:match_start].strip() == ""
    if at_start and idx > 0:
        return spans[idx - 1][0], match_end
    if at_start:
        return match_start, match_end
    return s, match_end


def _join_scratch(prefix, rest):
    """Glue the kept text. A continuation that now starts a sentence is capped.

    "Keep. " + "and continue." becomes "Keep. And continue." The same
    characters come back when there is nothing to join.
    """
    pre = prefix.rstrip()
    nxt = rest.lstrip()
    if not pre:
        if nxt and nxt[0].isalpha() and nxt[0].islower():
            return nxt[0].upper() + nxt[1:]
        return nxt
    if not nxt:
        return pre
    if pre[-1] in ".!?" and nxt[0].isalpha() and nxt[0].islower():
        nxt = nxt[0].upper() + nxt[1:]
    return pre + " " + nxt


def apply_voice_edits(text, cfg=None):
    """Drop a scratch-phrase and the sentence that ends before it.

    "scratch that." at the end of "Keep this. Drop me. Scratch that." removes
    "Drop me." and the command. "Keep. Drop. Scratch that and continue."
    removes "Drop." and the command, and caps the continuation:
    "Keep. And continue." The phrase alone, or with nothing before it,
    removes only itself. The same string comes back when nothing matches.
    """
    if not text:
        return text or ""
    phrases = _scratch_phrases(cfg)
    if not phrases:
        return text
    pat = _scratch_pattern(phrases)
    changed = False
    for _ in range(20):
        m = pat.search(text)
        if not m:
            break
        cut_from, cut_to = _scratch_cut(text, m.start(), m.end())
        if cut_to <= cut_from:
            cut_from, cut_to = m.start(), m.end()
        text = _join_scratch(text[:cut_from], text[cut_to:])
        changed = True
    return text.strip() if changed else text


def _drop_scratch_phrases(text, cfg=None):
    """Remove scratch phrases only. The surrounding sentences stay."""
    if not text:
        return text or ""
    phrases = _scratch_phrases(cfg)
    if not phrases:
        return text
    pat = _scratch_pattern(phrases)
    changed = False
    for _ in range(20):
        m = pat.search(text)
        if not m:
            break
        text = text[:m.start()] + text[m.end():]
        changed = True
    if not changed:
        return text
    return re.sub(r"[ \t]{2,}", " ", text)


def _glue_protected(protected, tail):
    """Join `tail` after `protected` without rewriting a single protected character."""
    if not protected:
        return tail or ""
    if not tail:
        return protected
    if protected[-1].isspace() or tail[0].isspace():
        return protected + tail
    return protected + " " + tail


def voice_edit_keeping_screen(text, typed_len, cfg=None):
    """Scratch-that for text that is not on screen yet.

    `typed_len` characters are already live-typed. They are returned
    unchanged — this never asks the caller to backspace. Each scratch
    command is judged on its own against that prefix. A command whose
    sentence is already inside the prefix drops only the phrase (when the
    phrase itself has not been typed) and the caller logs it. A later
    command that sits entirely past the prefix still deletes its own
    preceding sentence. Returns (new_text, typed_len).
    """
    text = text or ""
    typed_len = max(0, min(int(typed_len or 0), len(text)))
    if typed_len <= 0:
        return apply_voice_edits(text, cfg), 0
    phrases = _scratch_phrases(cfg)
    if not phrases:
        return text, typed_len
    pat = _scratch_pattern(phrases)
    protected = text[:typed_len]
    logged = False
    for _ in range(20):
        m = pat.search(text)
        if not m or m.end() <= typed_len:
            break
        if m.start() < typed_len:
            cut_from, cut_to = typed_len, m.end()
            eat_protected = True
        else:
            cut_from, cut_to = _scratch_cut(text, m.start(), m.end())
            if cut_to <= cut_from:
                cut_from, cut_to = m.start(), m.end()
            eat_protected = cut_from < typed_len
            if eat_protected:
                cut_from, cut_to = m.start(), m.end()
        if cut_to <= cut_from:
            break
        if eat_protected and not logged:
            print("[voice] scratch that left the on-screen sentence; dropped the phrase")
            logged = True
        nxt = _join_scratch(text[:cut_from], text[cut_to:])
        if not nxt.startswith(protected):
            tail = _join_scratch(text[typed_len:cut_from], text[cut_to:])
            nxt = _glue_protected(protected, tail)
        text = nxt
        typed_len = len(protected)
    return text, typed_len


def _paste_last_phrases(cfg=None):
    """Phrases that mean "paste the previous dictation".

    A config that sets the key, including to an empty list, wins. A missing
    key uses DEFAULTS.
    """
    block = None
    if isinstance(cfg, dict):
        block = cfg.get("voice_edits")
    if isinstance(block, dict) and "paste_last" in block:
        return [p for p in (block.get("paste_last") or []) if p]
    return list(DEFAULTS["voice_edits"]["paste_last"])


def normalize_voice_phrase(text):
    """Lower-case, drop punctuation, collapse whitespace.

    "Paste my last message!" and "paste my last message" compare equal.
    A hyphen is punctuation, so it is removed and does not become a space:
    "paste-my-last-message" does not match the command.
    """
    if not text:
        return ""
    kept = []
    for ch in str(text).lower():
        if ch.isalnum() or ch.isspace():
            kept.append(ch)
    return " ".join("".join(kept).split())


def match_paste_last(text, cfg=None):
    """True when the whole transcript is a paste-last command.

    The comparison is the normalized transcript against a normalized phrase.
    A longer sentence that only contains the phrase is not a command.
    """
    norm = normalize_voice_phrase(text)
    if not norm:
        return False
    phrases = {normalize_voice_phrase(p) for p in _paste_last_phrases(cfg)}
    return norm in phrases


def paste_last_held(text, cfg=None):
    """True when live typing must not put this commit on screen.

    The whole committed transcript is a paste-last command, or a word
    prefix of one ("paste my" before "paste my last message"). A longer
    sentence is ordinary text and may be typed. Holding the prefix is what
    keeps a short command off the screen. If any of it was typed anyway,
    the finished take is ordinary text — see apply_paste_last.
    """
    words = normalize_voice_phrase(text).split()
    if not words:
        return False
    for phrase in _paste_last_phrases(cfg):
        pw = normalize_voice_phrase(phrase).split()
        if not pw:
            continue
        if words == pw or (len(words) < len(pw) and words == pw[:len(words)]):
            return True
    return False


def apply_paste_last(text, live_typed, last_text="", cfg=None):
    """Decide whether this finished take pastes the previous dictation.

    Rule: the command counts only when the whole final transcript matches
    and nothing from this take was live-typed (`live_typed` == 0). A
    live-typed commit leaves those words on screen, so the take is ordinary
    dictation and nothing is deleted. The spoken phrase is `history`. It
    does not replace `last_text`. An empty previous dictation sets `empty`
    and pastes nothing.

    `text` is the transcript after strip_hallucinations and scratch-that.
    That is the text which would otherwise be pasted.
    """
    text = text or ""
    last_text = last_text or ""
    typed = int(live_typed or 0)
    if typed > 0 or not match_paste_last(text, cfg):
        return {
            "kind": "text",
            "paste": text,
            "history": text,
            "last_text": text or last_text,
            "empty": False,
        }
    return {
        "kind": "command",
        "paste": last_text,
        "history": text,
        "last_text": last_text,
        "empty": not bool(last_text),
    }


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
        self._buf_lock = threading.Lock()
        # Bumped on start and again when the take is sealed. discard_prefix
        # and snapshot refuse a generation that is no longer current, so a
        # streamer from the previous take cannot trim or read the next one.
        self._gen = 0
        # Samples dropped from the front of this take. The text for them is
        # already committed; snapshot/stop only carry the tail.
        self._discarded = 0
        self.last_full_samples = 0
        self.last_discarded = 0
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
        with self._buf_lock:
            if self._recording:
                self._chunks.append(chunk)
            else:
                self._ring.append(chunk)
                self._ring_samples += len(chunk)
                while (self._ring and
                       self._ring_samples - len(self._ring[0]) >= self.pre_roll):
                    self._ring_samples -= len(self._ring.popleft())

    def start(self, prefix=None):
        """Begin a take. `prefix` seeds the buffer and skips the pre-roll ring.

        Returns this take's generation. A hands-free latch does not restart
        the recorder and does not pass a sealed prefix.
        """
        if self._stream is None or not self._stream.active:
            self._open()  # recover from device loss
        with self._buf_lock:
            self._gen += 1
            if prefix is not None and len(prefix):
                arr = np.asarray(prefix, dtype=np.float32).reshape(-1, 1)
                self._chunks = [arr]
            else:
                # pre-roll: capture speech before the press
                self._chunks = list(self._ring)
            self._discarded = 0
            self._recording = True
            return self._gen

    def generation(self):
        with self._buf_lock:
            return self._gen

    def snapshot(self, gen=None):
        """Audio still held (recording continues). O(uncommitted) after a trim.

        A `gen` from a take that has already been sealed returns empty, so a
        stale streamer cannot read the next take.
        """
        with self._buf_lock:
            if gen is not None and gen != self._gen:
                return np.zeros(0, dtype=np.float32)
            chunks = list(self._chunks)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).flatten()

    def discard_prefix(self, n, gen=None, owner=None):
        """Drop the first n samples and rebase `owner.committed_samples`.

        n indexes snapshot(), not the original take. The text for this audio
        is already committed; keeping the samples would make every later pass
        concatenate the whole dictation. Returns how many samples were dropped.
        A `gen` that is not the current take drops nothing.

        The trim and the owner's offset move under this lock, before it is
        released. A capture that runs as soon as the lock is free therefore
        sees the shortened buffer and the shortened offset together — never
        the new buffer with the old offset.
        """
        n = int(n)
        if n <= 0:
            return 0
        with self._buf_lock:
            if gen is not None and gen != self._gen:
                return 0
            if owner is not None and getattr(owner, "_frozen", None) is not None:
                return 0
            dropped = 0
            while n > 0 and self._chunks:
                chunk = self._chunks[0]
                frames = int(chunk.shape[0])
                if frames <= n:
                    self._chunks.pop(0)
                    n -= frames
                    dropped += frames
                else:
                    self._chunks[0] = chunk[n:]
                    dropped += n
                    n = 0
            self._discarded += dropped
            if owner is not None and dropped:
                cur = int(getattr(owner, "committed_samples", 0) or 0)
                owner.committed_samples = cur - dropped
        return dropped

    def capture(self, streamer=None):
        """Seal the take. Return (audio, (committed_text, committed_samples)).

        The copy and the commit offsets are taken under one lock, then the
        generation advances. A streamer still inside a pass cannot trim this
        buffer or the next take, and it cannot rebase the offsets just read.
        """
        with self._buf_lock:
            self._recording = False
            chunks = list(self._chunks)
            discarded = int(self._discarded)
            self._discarded = 0
            frames = sum(int(c.shape[0]) for c in chunks)
            if streamer is not None:
                frozen = (getattr(streamer, "committed_text", "") or "",
                          int(getattr(streamer, "committed_samples", 0) or 0))
                try:
                    streamer._frozen = frozen
                    # Same lock as the edited text. A later scratch cannot
                    # rewrite the transcript history stores as raw.
                    if hasattr(streamer, "committed_raw"):
                        streamer._frozen_raw = streamer.committed_raw
                except Exception:
                    pass
            else:
                frozen = ("", 0)
            self.last_discarded = discarded
            self.last_full_samples = discarded + frames
            self._gen += 1
        audio = (np.concatenate(chunks).flatten() if chunks
                 else np.zeros(0, dtype=np.float32))
        return audio, frozen

    def stop(self):
        """End the take and return its audio. See capture()."""
        audio, _frozen = self.capture(None)
        return audio

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

    def __init__(self, flow, live=False, source=None, voice_edits=True):
        self.flow = flow
        self.source = source or flow.recorder  # anything with .snapshot()
        self.committed_text = ""
        # Phrases before scratch-that. Raw history freezes this at release.
        # `committed_text` is the edited transcript.
        self.committed_raw = ""
        self.committed_samples = 0
        self.live = live         # paste committed words while still recording
        self.typed_len = 0       # chars of committed_text already pasted
        self.typed_chars = 0     # chars actually put on screen (after replacements)
        self.published_len = 0   # chars already handed to a remote client
        # Remote StreamSession workers pass False. A thin client has already
        # typed the published prefix; editing it makes the client's delta
        # slice corrupt ("Drop me." + "s."). Local hands-free keeps edits.
        self.voice_edits = bool(voice_edits)
        self._gen = None         # Recorder generation this worker may trim
        self._frozen = None      # (text, samples) sealed with the audio
        self.saved_clip = None
        # Generation at the moment this worker saved the clipboard. Restores
        # use this value. A later Flow write, including our own paste, moves
        # it; cleanup must not read a fresh generation.
        self.clip_gen = _clipboard_generation()
        self._prev = []          # word texts from the previous pass
        self._active = False
        self._thread = None
        self._wake = threading.Event()  # cuts the idle wait; does not abort a pass
        if live:
            try:
                self.saved_clip = pyperclip.paste()
            except Exception:
                pass
            self.clip_gen = _clipboard_generation()

    def start(self):
        # A previous stop() leaves the event set. Clear it before the thread
        # exists so the first wait is a full interval, same as the old sleep.
        self._wake.clear()
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def halt(self):
        """Stop committing. Join happens later, off the ctrl-loop thread."""
        self._active = False
        self._wake.set()

    def stop(self):
        self._active = False
        self._wake.set()
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
            self._wake.wait(interval)
            if not self._active:
                return
            if self.flow.model_lock.locked():
                continue  # don't queue behind other GPU work; try next cycle
            snap = self._snapshot()
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
            if not self._generation_ok():
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
                # Live text is typed before the final pass, so loops must be
                # caught here too. Filter the joined text; when the loop runs
                # into what is already on screen, keep only the new remainder.
                joined = (self.committed_text + " " + phrase).strip()
                clean = strip_hallucinations(joined)
                if clean != joined:
                    base = self.committed_text
                    phrase = (clean[len(base):].strip()
                              if clean.startswith(base) else "")
                    print(f"[stream] dropped a looped phrase")
                self._commit_phrase(phrase, int(words[cut - 1].end * sr),
                                    texts[cut:])
            else:
                self._prev = texts

    def _generation_ok(self):
        gen = getattr(self, "_gen", None)
        src_gen = getattr(self.source, "_gen", None)
        if gen is None or src_gen is None:
            return True
        return gen == src_gen

    def _snapshot(self):
        snap = self.source.snapshot
        gen = getattr(self, "_gen", None)
        if gen is None:
            return snap()
        try:
            return snap(gen)
        except TypeError:
            return snap()

    def _should_live_type(self):
        """Hands-free and rewrite never type while the take is open.

        A latched take commits in memory and pastes once at the end, so
        scratch-that can edit pure text and a moved caret is never backspaced.
        """
        if not self.live or not self._active:
            return False
        if getattr(self, "_frozen", None) is not None:
            return False
        flow = self.flow
        if getattr(flow, "rewrite_take", False) or getattr(flow, "latched", False):
            return False
        return True

    def _commit_phrase(self, phrase, sample_at, rest_prev):
        """Commit under the recorder lock, then type and trim outside it.

        capture() holds the same lock while it copies the audio and the
        offsets. A pass that loses the race sees the sealed generation and
        leaves both alone.
        """
        lock = getattr(self.source, "_buf_lock", None)

        def apply():
            if not self._active or getattr(self, "_frozen", None) is not None:
                return False
            if not self._generation_ok():
                return False
            self.committed_raw = (
                getattr(self, "committed_raw", "") + " " + phrase).strip()
            self.committed_text = (self.committed_text + " " + phrase).strip()
            self.committed_samples += int(sample_at)
            self._prev = rest_prev
            if getattr(self, "voice_edits", True):
                protected = int(self.typed_len or 0)
                if not self.live:
                    # Hands-free on this PC has published nothing, so this
                    # stays 0. Remote sessions do not edit at all.
                    protected = max(
                        protected, int(getattr(self, "published_len", 0) or 0))
                if protected > len(self.committed_text):
                    protected = len(self.committed_text)
                edited, _kept = voice_edit_keeping_screen(
                    self.committed_text, protected, self.flow.cfg)
                self.committed_text = edited
                pub = int(getattr(self, "published_len", 0) or 0)
                if pub > len(self.committed_text):
                    self.published_len = len(self.committed_text)
            return True

        if lock is None:
            ok = apply()
        else:
            with lock:
                ok = apply()
        if not ok:
            return
        if self._should_live_type():
            self.type_chunk()
        self._drop_committed_audio()

    def _drop_committed_audio(self):
        """Drop samples whose text is committed, and typed when live-typing.

        committed_samples is rebased onto what snapshot() still returns, so
        the final `audio[committed_samples:]` stays the uncommitted tail.
        A sealed or stale generation drops nothing and does not rebase.
        """
        if self.committed_samples <= 0:
            return
        if getattr(self, "_frozen", None) is not None:
            return
        if not self._generation_ok():
            return
        # A paste-last command is held off the screen on purpose. Drop its
        # audio anyway so the final decode does not transcribe it twice.
        # Any other untyped live commit stays in the buffer until it lands.
        held = (getattr(self, "voice_edits", True)
                and paste_last_held(
                    self.committed_text, getattr(self.flow, "cfg", None)))
        if self.live and self.typed_len < len(self.committed_text) and not held:
            return
        drop = getattr(self.source, "discard_prefix", None)
        if drop is None:
            return
        gen = getattr(self, "_gen", None)
        # The recorder rebases this worker inside the trim lock. A fake that
        # only accepts (n) or (n, gen) still rebases here, after it returns.
        rebased = False
        try:
            dropped = drop(self.committed_samples, gen, self)
            rebased = True
        except TypeError:
            try:
                dropped = (drop(self.committed_samples) if gen is None
                           else drop(self.committed_samples, gen))
            except TypeError:
                dropped = drop(self.committed_samples)
        if dropped and not rebased:
            self.committed_samples -= dropped

    def note_published(self, n):
        """The first `n` characters have been sent to a client that may type them."""
        n = max(0, int(n or 0))
        if n > int(getattr(self, "published_len", 0) or 0):
            self.published_len = n

    def type_chunk(self):
        """Paste newly committed words at the cursor while still recording.
        Uses clipboard+ctrl+v: the user is physically holding ctrl already, so
        synthetic character typing would trigger app shortcuts — paste is safe."""
        if not self._should_live_type():
            return
        start = self.typed_len
        chunk = self.committed_text[start:]
        if not chunk:
            return
        # The phrase is still only a paste-last command, or a word prefix of
        # one. Leave it untyped. If a later commit grows past the command,
        # that commit types from here. If this commit was typed anyway, the
        # finished take is ordinary text and is not a paste-last command.
        if (getattr(self, "voice_edits", True)
                and paste_last_held(
                    self.committed_text, getattr(self.flow, "cfg", None))):
            return
        try:
            pasted = apply_replacements(chunk, self.flow.cfg)
            self._paste_screen(pasted)
            self.typed_len = len(self.committed_text)
            self.typed_chars += len(pasted)
        except Exception as e:
            print(f"[stream] live type failed: {e}")

    def _paste_screen(self, text):
        """Paste text that is already in screen form (replacements applied).

        The copy goes through the shared clipboard writer so a delayed
        restore from an older take sees the generation move and does nothing.
        """
        if not text:
            return
        written = _copy_clipboard(text)
        if not written:
            raise RuntimeError("clipboard copy failed")
        self.clip_gen = (written if type(written) is int
                         else _clipboard_generation())
        time.sleep(0.02)
        keyboard.send("ctrl+v")

    def restore_clipboard(self, expect_gen=None):
        """Put the user's clipboard back unless a newer Flow write owns it.

        `expect_gen` is the generation recorded when this clipboard was saved
        or last written by this worker. A missing argument uses that recorded
        value. Cleanup must not substitute the generation it sees now.
        """
        if self.saved_clip is None:
            return
        if expect_gen is None:
            expect_gen = getattr(self, "clip_gen", None)
        if expect_gen is None:
            return
        wrote = _copy_clipboard(self.saved_clip, expect_gen=expect_gen)
        if wrote:
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
        self.clip_gen = _clipboard_generation()
        self._active = False
        self._thread = None
        self._wake = threading.Event()

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
        self.clip_gen = _clipboard_generation()
        self._wake.clear()
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import requests
        sr = self.flow.cfg["sample_rate"]
        while self._active:
            self._wake.wait(0.35)
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
        # Same hold as the local streamer: a paste-last command stays off
        # the screen until the take proves it is a longer sentence.
        if paste_last_held(self.committed, getattr(self.flow, "cfg", None)):
            return
        try:
            written = _copy_clipboard(chunk)
            if not written:
                raise RuntimeError("clipboard copy failed")
            self.clip_gen = (written if type(written) is int
                             else _clipboard_generation())
            time.sleep(0.02)
            keyboard.send("ctrl+v")  # user is already holding ctrl — paste is safe
            self.typed = len(self.committed)
        except Exception as e:
            print(f"[client-stream] live type failed: {e}")

    def halt(self):
        """Stop uploading. finish() joins the thread off the ctrl loop."""
        self._active = False
        self._wake.set()

    def stop(self):
        """on_release calls this like a StreamWorker; the real work is finish()."""
        self.halt()
        self._wake.set()
        if self._thread:
            self._thread.join(2.0)
        return self.committed, 0

    def finish(self):
        """Stop uploading; final decode on the server. Returns (text, typed)."""
        import requests
        self._active = False
        self._wake.set()
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

    def restore_clipboard(self, expect_gen=None):
        """See StreamWorker.restore_clipboard. A stale generation does not copy."""
        if self.saved_clip is None:
            return
        if expect_gen is None:
            expect_gen = getattr(self, "clip_gen", None)
        if expect_gen is None:
            return
        _copy_clipboard(self.saved_clip, expect_gen=expect_gen)


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

    def __init__(self, level_fn=None, hints=True, hint_line=None):
        import queue
        self._q = queue.Queue()
        self._empty = queue.Empty
        self._level_fn = level_fn or (lambda: 0.0)
        self.hints = hints
        self._hint_line = hint_line
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
        # Packed only while a plain recording wants the hint line. pack_forget
        # otherwise, so summary/rewrite/hands-free don't grow the pill.
        hint_fg = "#6e7380"
        hints_lbl = tk.Label(root, text="", fg=hint_fg, bg=bg,
                             font=("Segoe UI", 8), anchor="w", justify="left")

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

        def hide_hints():
            if hints_lbl.winfo_ismapped():
                hints_lbl.pack_forget()
                context.pack_configure(pady=(0, 8))

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
                        flag = (s[2] if isinstance(s, tuple) and len(s) > 2
                                else None)
                        if flag == "handsfree":
                            raw_ctx = (s[1] if isinstance(s, tuple) and len(s) > 1
                                       and s[1] else "00:00")
                        elif flag in ("summary", "rewrite"):
                            raw_ctx = ""
                        else:
                            raw_ctx = describe_target(
                                s[1] if isinstance(s, tuple) and len(s) > 1
                                else "")
                        status_text, ctx_text, hint_text = rec_overlay(
                            flag, raw_ctx, self.hints, self._hint_line)
                        status.config(text=status_text)
                        context.config(text=ctx_text)
                        if hint_text:
                            # The status row sets the width. Clip the hint to
                            # that, or a long line widens the pill off a narrow
                            # window. place() itself is unchanged.
                            top.update_idletasks()
                            px = max(40, top.winfo_reqwidth() - 28)
                            # 7.2px/char overestimates Segoe UI 8, so the line
                            # is clipped short of the pill rather than widening it.
                            hints_lbl.config(
                                text=clip_line(hint_text, max(12, int(px / 7.2))))
                            if not hints_lbl.winfo_ismapped():
                                context.pack_configure(pady=(0, 1))
                                hints_lbl.pack(fill="x", padx=15, pady=(0, 8))
                        elif hints_lbl.winfo_ismapped():
                            hints_lbl.pack_forget()
                            context.pack_configure(pady=(0, 8))
                        hide_at[0] = None
                        show("rec")
                        place()
                    elif kind == "live":
                        hide_hints()
                        status.config(text="Live listening")
                        context.config(text="Say the wake word, then a command")
                        hide_at[0] = None
                        show("live")
                        place()
                    elif kind == "work":
                        hide_hints()
                        label = "Transcribing…"
                        if isinstance(s, tuple) and len(s) > 1 and s[1]:
                            label = str(s[1])[:200]
                        status.config(text=label)
                        context.config(text="")
                        hide_at[0] = None
                        show("work")
                        place()
                    elif kind == "say":
                        hide_hints()
                        status.config(text=s[1][:200])
                        context.config(text="")
                        show("say")
                        place()
                        hide_at[0] = time.time() + max(4, min(12, len(s[1]) / 18))
                    elif kind == "ignored":
                        hide_hints()
                        heard = (s[1] or "")[:40]
                        status.config(text=heard or "Heard something", fg=sub)
                        context.config(text="say 'flow' first", fg=sub)
                        show("live")
                        place()
                        hide_at[0] = time.time() + 1.5
                        revert_mode[0] = "live"
                    else:  # idle
                        hide_hints()
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


SUMMARY_PROMPT = (
    "You rewrite a dictated voice note so it is to the point. Keep every "
    "concrete fact, number, name, question and request; drop filler, "
    "repetition, false starts and thinking out loud. Keep the speaker's voice "
    "(first person, same intent: a question stays a question, an instruction "
    "stays an instruction). Use short sentences, or a few '- ' bullets only if "
    "there are several separate points. Never add facts, greetings or "
    "commentary. Output only the rewritten text."
)

SUMMARY_DEFAULTS = [
    # multivac's 4080 via the flow-asr node's token-checked /summarize proxy,
    # so beast's GPU stays free and multivac's Ollama stays on loopback.
    {"provider": "flow-node", "model": "qwen3.5:9b",
     "url": "http://multivac:8757/summarize"},
    # beast fallback: qwen3:8b (5 GB) fits beside Whisper on the 5090.
    {"provider": "ollama", "model": "qwen3:8b",
     "base_url": "http://127.0.0.1:11434"},
]


REWRITE_PROMPT = (
    "You rewrite the text the user selected, following their spoken instruction. "
    "Change only what the instruction asks for. Preserve meaning the instruction "
    "does not change, and keep names, numbers and wording it does not mention. "
    "Output only the rewritten text — no quotes, no preamble, no explanation."
)


def _summary_call(p, text, cfg, system=None):
    """One summary-lane call. `system` defaults to the to-the-point prompt
    so rewrite can reuse the same lanes without a second client."""
    system = SUMMARY_PROMPT if system is None else system
    if p.get("provider") == "flow-node":
        r = requests.post(
            p["url"], timeout=p.get("timeout_s", 30),
            headers={"Authorization": "Bearer " + cfg["server"]["token"]},
            json={"model": p.get("model"), "system": system, "text": text})
        r.raise_for_status()
        out = (r.json().get("text") or "").strip()
        if not out:
            raise ValueError("empty response")
        return out
    return llm.call_provider(p, system, text, max_tokens=1024)


def llm_summarize(text, cfg, system=None):
    """'To the point' rewrite for a shift+dictation take. None on failure,
    so the caller pastes the raw transcript instead of nothing. Lanes are the
    user's own machines (config "summary.lanes"), so local_only does not apply.
    `system` overrides SUMMARY_PROMPT (rewrite selection)."""
    lanes = (cfg.get("summary") or {}).get("lanes") or SUMMARY_DEFAULTS
    for p in lanes:
        t0 = time.time()
        try:
            out = _summary_call(p, text, cfg, system)
            print(f"[summary] {p.get('model')} in {time.time() - t0:.2f}s "
                  f"({len(text.split())} -> {len(out.split())} words)")
            return out
        except Exception as e:
            print(f"[summary] {p.get('model')} failed: {e}")
    return None


def llm_rewrite(instruction, selection, cfg):
    """Rewrite `selection` per the spoken instruction. None on failure."""
    body = f"Instruction: {instruction}\n\nText:\n{selection}"
    return llm_summarize(body, cfg, system=REWRITE_PROMPT)


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


def foreground_hwnd():
    """HWND of the foreground window, or 0 when it cannot be read."""
    try:
        import ctypes
        return int(ctypes.windll.user32.GetForegroundWindow() or 0)
    except Exception:
        return 0


def focused_control_hwnd():
    """Focused control in the foreground thread, or 0 when it cannot be read.

    GetGUIThreadInfo(foreground thread).hwndFocus. Tests monkeypatch this.
    0 means the focused control could not be identified.
    """
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        fg = user32.GetForegroundWindow()
        if not fg:
            return 0
        tid = user32.GetWindowThreadProcessId(fg, None)

        class RECT(ctypes.Structure):
            _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                        ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

        class GUITHREADINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hwndActive", wintypes.HWND),
                ("hwndFocus", wintypes.HWND),
                ("hwndCapture", wintypes.HWND),
                ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND),
                ("hwndCaret", wintypes.HWND),
                ("rcCaret", RECT),
            ]

        info = GUITHREADINFO()
        info.cbSize = ctypes.sizeof(GUITHREADINFO)
        if not user32.GetGUIThreadInfo(int(tid), ctypes.byref(info)):
            return 0
        return int(info.hwndFocus or 0)
    except Exception:
        return 0


# WM_GETTEXTLENGTH, EM_GETSEL, EM_REPLACESEL. Quiet insert verifies the
# length change. EM_GETSEL supplies the selection the length delta replaces.
_WM_GETTEXTLENGTH = 0x000E
_EM_GETSEL = 0x00B0
_EM_REPLACESEL = 0x00C2
_VK_MENU = 0x12
_KEYEVENTF_KEYUP = 0x0002
# Focus switch plus this settle is the sticky budget. The modifier wait is
# the same 1.5s cap as summary paste and is not part of this budget.
_STICKY_SETTLE_S = 0.06
STICKY_UNAVAILABLE = "Copied — the box you started in isn't available"
PASTE_LAST_EMPTY = "No dictation to paste yet"


def sticky_enabled(cfg):
    """True unless this config turns sticky paste off.

    A missing key uses DEFAULTS, which ships on.
    """
    if not isinstance(cfg, dict) or "sticky_target" not in cfg:
        return bool(DEFAULTS.get("sticky_target", True))
    return bool(cfg.get("sticky_target"))


def classic_edit_class(name):
    """Win32 edit controls that accept EM_REPLACESEL.

    Exact "Edit", or a RichEdit / RICHEDIT50W class. Anything else, including
    a browser renderer, is not a quiet-insert target.
    """
    if not name or not isinstance(name, str):
        return False
    if name == "Edit":
        return True
    return name.upper().startswith("RICHEDIT")


def _wchar_len(text):
    """UTF-16 code units. WM_GETTEXTLENGTH counts those, not Unicode scalars."""
    return len((text or "").encode("utf-16-le")) // 2


def unpack_take_target(take_hwnd, take_focus=0):
    """Foreground HWND and focused control captured when the take opened.

    The job tuple stores them as one pair so the release HWND stays the last
    argument. A bare integer is an older caller that did not capture focus.
    """
    if isinstance(take_hwnd, (tuple, list)):
        hwnd = int(take_hwnd[0] or 0) if take_hwnd else 0
        focus = int(take_hwnd[1] or 0) if len(take_hwnd) > 1 else int(take_focus or 0)
        return hwnd, focus
    return int(take_hwnd or 0), int(take_focus or 0)


class TargetBridge:
    """Win32 calls sticky paste needs. Tests replace the instance.

    Same foreground window, different browser tab or field: the HWND does
    not change, and nothing here can see the tab. Callers paste as today.
    """

    def __init__(self):
        self._user32 = None
        self._kernel32 = None

    def _libs(self):
        if self._user32 is None:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            user32.SendMessageW.argtypes = [
                wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            user32.SendMessageW.restype = ctypes.c_ssize_t
            user32.IsWindow.argtypes = [wintypes.HWND]
            user32.IsWindow.restype = wintypes.BOOL
            user32.IsIconic.argtypes = [wintypes.HWND]
            user32.IsIconic.restype = wintypes.BOOL
            user32.SetForegroundWindow.argtypes = [wintypes.HWND]
            user32.SetForegroundWindow.restype = wintypes.BOOL
            user32.SetFocus.argtypes = [wintypes.HWND]
            user32.SetFocus.restype = wintypes.HWND
            user32.GetClassNameW.argtypes = [
                wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
            user32.GetClassNameW.restype = ctypes.c_int
            user32.AttachThreadInput.argtypes = [
                wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
            user32.AttachThreadInput.restype = wintypes.BOOL
            user32.GetWindowThreadProcessId.argtypes = [
                wintypes.HWND, ctypes.c_void_p]
            user32.GetWindowThreadProcessId.restype = wintypes.DWORD
            user32.keybd_event.argtypes = [
                wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_ulonglong]
            kernel32.GetCurrentThreadId.restype = wintypes.DWORD
            self._user32 = user32
            self._kernel32 = kernel32
        return self._user32, self._kernel32

    def foreground(self):
        """Foreground HWND. Goes through foreground_hwnd so tests can patch it."""
        return int(foreground_hwnd() or 0)

    def class_name(self, hwnd):
        try:
            user32, _kernel = self._libs()
            buf = __import__("ctypes").create_unicode_buffer(256)
            n = user32.GetClassNameW(int(hwnd), buf, 256)
            if not n:
                return ""
            return buf.value
        except Exception:
            return ""

    def is_window(self, hwnd):
        hwnd = int(hwnd or 0)
        if hwnd <= 0:
            return False
        try:
            user32, _kernel = self._libs()
            return bool(user32.IsWindow(hwnd))
        except Exception:
            return False

    def is_minimized(self, hwnd):
        hwnd = int(hwnd or 0)
        if hwnd <= 0:
            return False
        try:
            user32, _kernel = self._libs()
            return bool(user32.IsIconic(hwnd))
        except Exception:
            return False

    def text_length(self, hwnd):
        """WM_GETTEXTLENGTH, or None when the control cannot be read."""
        hwnd = int(hwnd or 0)
        if hwnd <= 0 or not self.is_window(hwnd):
            return None
        try:
            user32, _kernel = self._libs()
            return int(user32.SendMessageW(hwnd, _WM_GETTEXTLENGTH, 0, 0))
        except Exception:
            return None

    def selection_span(self, hwnd):
        """EM_GETSEL as (start, end), or None when it cannot be read."""
        hwnd = int(hwnd or 0)
        if hwnd <= 0:
            return None
        try:
            import ctypes
            user32, _kernel = self._libs()
            start = ctypes.c_ulong()
            end = ctypes.c_ulong()
            user32.SendMessageW(
                hwnd, _EM_GETSEL,
                ctypes.addressof(start), ctypes.addressof(end))
            return int(start.value), int(end.value)
        except Exception:
            return None

    def replace_selection(self, hwnd, text):
        """EM_REPLACESEL with undo. False when the message was not sent."""
        hwnd = int(hwnd or 0)
        if hwnd <= 0:
            return False
        try:
            import ctypes
            user32, _kernel = self._libs()
            buf = ctypes.create_unicode_buffer(text or "")
            user32.SendMessageW(
                hwnd, _EM_REPLACESEL, 1, ctypes.addressof(buf))
            return True
        except Exception:
            return False

    def force_foreground(self, hwnd, focus=0):
        """Bring `hwnd` forward and focus the captured control.

        AttachThreadInput, then SetForegroundWindow. If the foreground is
        still not `hwnd`, tap ALT once and try SetForegroundWindow again.
        SetFocus targets the captured control. True only when
        GetForegroundWindow() is `hwnd` afterwards.
        """
        hwnd = int(hwnd or 0)
        focus = int(focus or 0)
        if hwnd <= 0:
            return False
        user32 = kernel32 = None
        attached = []
        try:
            user32, kernel32 = self._libs()
            my_tid = int(kernel32.GetCurrentThreadId() or 0)
            current = self.foreground()
            fg_tid = 0
            if current:
                fg_tid = int(user32.GetWindowThreadProcessId(current, None) or 0)
            tgt_tid = int(user32.GetWindowThreadProcessId(hwnd, None) or 0)

            def attach(a, b):
                if a and b and a != b and user32.AttachThreadInput(a, b, True):
                    attached.append((a, b))

            attach(my_tid, fg_tid)
            attach(my_tid, tgt_tid)
            user32.SetForegroundWindow(hwnd)
            if self.foreground() != hwnd:
                user32.keybd_event(_VK_MENU, 0, 0, 0)
                user32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)
                user32.SetForegroundWindow(hwnd)
            if focus:
                user32.SetFocus(focus)
            return self.foreground() == hwnd
        except Exception:
            return False
        finally:
            if user32 is not None:
                for a, b in attached:
                    try:
                        user32.AttachThreadInput(a, b, False)
                    except Exception:
                        pass


def quiet_insert(bridge, hwnd, text):
    """Insert `text` with EM_REPLACESEL. No focus change.

    Success is WM_GETTEXTLENGTH moving by len(text) minus the selection.
    A length that does not move means the control ignored the message.
    A length that moves by some other amount is kept, so a second paste
    cannot write the dictation twice. Returns False without sending when
    the length or the selection cannot be read.
    """
    if not text or not hwnd:
        return False
    before = bridge.text_length(hwnd)
    if before is None:
        return False
    span = None
    getter = getattr(bridge, "selection_span", None)
    if not getter:
        return False
    try:
        span = getter(hwnd)
    except Exception:
        span = None
    if not span or len(span) != 2:
        return False
    if span[0] is None or span[1] is None:
        return False
    sel = max(0, int(span[1]) - int(span[0]))
    if sel > int(before):
        return False
    try:
        sent = bridge.replace_selection(hwnd, text)
    except Exception:
        return False
    if sent is False:
        return False
    after = bridge.text_length(hwnd)
    if after is None:
        return False
    expected = int(before) - sel + _wchar_len(text)
    if int(after) == expected:
        return True
    return int(after) != int(before)


def sticky_deliver(text, target, bridge, paste, copy, *,
                   enabled=True, on_current=None, wait_modifiers=None,
                   settle=None):
    """Paste `text` into the window captured at press.

    Same foreground HWND: call `paste` (clipboard or typed paste, as today).
    A different tab or field in that same window is not detectable. Nothing
    extra happens.

    A different window tries a quiet EM_REPLACESEL into a classic edit
    (Edit / RichEdit* / RICHEDIT50W). If that fails, and the original
    window still exists and is not minimized, switch back, confirm the
    foreground, wait until ctrl/alt/shift are up, paste, settle ~60ms, then
    restore the user's window. A failed confirm pastes nowhere, copies, and
    returns STICKY_UNAVAILABLE.

    A restore that fails after a confirmed paste leaves the text in the
    original box. The paste already happened, so it is not copied again.

    `on_current` runs only while the foreground still matches. Return
    "paste", or ("copy", message). Summary uses it for modifiers and for
    input since release. Window changes are handled here.

    `paste` returns a clipboard generation or None. `copy` is truthy when
    the copy landed. No captured HWND pastes as today: a dictation is not
    dropped because the target was never read.
    """
    def _current(generation=None):
        return {
            "action": "current", "pasted": True, "copied": False,
            "message": None, "generation": generation,
        }

    def _copied(message, copied):
        return {
            "action": "copy", "pasted": False, "copied": bool(copied),
            "message": message,
            "generation": None,
        }

    if not enabled:
        return _current(paste())
    hwnd = int((target or {}).get("hwnd") or 0)
    focus = int((target or {}).get("focus") or 0)
    if not hwnd:
        return _current(paste())
    now = int(bridge.foreground() or 0)
    if now == hwnd:
        if on_current is not None:
            decision = on_current()
            if decision != "paste":
                _kind, message = decision
                return _copied(message, copy())
        return _current(paste())

    if focus and classic_edit_class(bridge.class_name(focus)):
        if quiet_insert(bridge, focus, text):
            return {
                "action": "quiet", "pasted": True, "copied": False,
                "message": None, "generation": None,
            }

    user = now
    can_switch = bridge.is_window(hwnd) and not bridge.is_minimized(hwnd)
    switched = False
    if can_switch:
        try:
            brought = bridge.force_foreground(hwnd, focus)
            switched = brought and int(bridge.foreground() or 0) == hwnd
            if switched:
                if wait_modifiers is None:
                    wait_modifiers = wait_modifiers_up
                if not wait_modifiers():
                    switched = False
                else:
                    generation = paste()
                    if settle is None:
                        time.sleep(_STICKY_SETTLE_S)
                    else:
                        settle()
                    return {
                        "action": "switch", "pasted": True, "copied": False,
                        "message": None, "generation": generation,
                    }
        finally:
            if user and user != hwnd and int(bridge.foreground() or 0) != user:
                try:
                    bridge.force_foreground(user, 0)
                except Exception:
                    pass
    return _copied(STICKY_UNAVAILABLE, copy())


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


_SUMMARY_KEYS = ("shift", "left shift", "right shift")
_REWRITE_KEYS = ("alt", "left alt", "right alt")
# Double-tap latch. There is no deferred release. Every ordinary right-ctrl
# release is processed immediately. A press that begins within _LATCH_GAP_S
# of a previous press shorter than _LATCH_MAX_PRESS_S is a latch candidate.
# If that candidate is itself released before _LATCH_MAX_PRESS_S, it becomes
# hands-free and keeps recording. A longer second press ends at its release.
# While latched, the next right-ctrl down ends the take; its up is ignored.
# Client mode does not latch.
_LATCH_MAX_PRESS_S = 0.35
_LATCH_GAP_S = 0.40
_LATCH_IDLE = {
    "phase": "idle", "down_at": None, "candidate": False,
    "prev_up_at": None, "prev_press_s": None,
}
# VK_CONTROL, VK_MENU, VK_SHIFT. Either side counts.
_MODIFIER_VKS = (0x11, 0x12, 0x10)
_MODIFIER_WAIT_S = 1.5


def _transcribing_label(elapsed):
    """Overlay line for a decode that has already run past _WAIT_HINT_S."""
    whole = max(1, int(elapsed))
    if elapsed >= _WAIT_BUSY_S:
        return f"Still transcribing ({whole}s) — the PC is busy"
    return f"Transcribing… ({whole}s)"


def _rewrite_enabled(cfg):
    """Selection rewrite is off unless this config opts in.

    A missing block uses DEFAULTS, which ships disabled.
    """
    if not isinstance(cfg, dict):
        cfg = DEFAULTS
    block = cfg.get("rewrite")
    if not isinstance(block, dict):
        block = DEFAULTS.get("rewrite") or {}
    return bool(block.get("enabled", False))


def overlay_hint_line(cfg=None):
    """Third pill line. 'Alt rewrite' is absent while rewrite is off."""
    rewrite = "Alt rewrite · " if _rewrite_enabled(cfg) else ""
    return (
        "⇧ to the point · "
        + rewrite
        + "double-tap hands-free · \"scratch that\""
    )


_HINT_LINE = overlay_hint_line(None)


def _overlay_flags(cfg):
    """(enabled, hints). A legacy bool config means the pill is on, hints on."""
    raw = cfg.get("overlay", True) if isinstance(cfg, dict) else True
    if isinstance(raw, dict):
        return bool(raw.get("enabled", True)), bool(raw.get("hints", True))
    return bool(raw), True


def clip_line(text, max_chars):
    """One line. Narrow pills keep the start and mark the cut."""
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return "…"
    return text[:max_chars - 1] + "…"


def format_elapsed(seconds):
    whole = max(0, int(seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def format_latched_context(seconds, words):
    """Hands-free context line: elapsed time and how many words are committed."""
    n = max(0, int(words or 0))
    unit = "word" if n == 1 else "words"
    return f"{format_elapsed(seconds)} · {n} {unit}"


def rec_overlay(flag, context, hints_enabled, hint_line=None):
    """Status, context, hint for the recording pill.

    The hint is a plain hold-to-talk affordance. Summary, rewrite and
    hands-free already use the status line, so a third line there would
    only crowd a narrow window. `hint_line` None is the disabled default,
    which does not mention Alt rewrite.
    """
    if flag == "summary":
        return ("Listening · to the point", "Summarizing when you let go", "")
    if flag == "rewrite":
        return ("Listening · rewrite selection", "", "")
    if flag == "handsfree":
        return ("Hands-free · tap right ctrl to stop", context or "00:00", "")
    line = _HINT_LINE if hint_line is None else hint_line
    return ("Listening", context or "", line if hints_enabled else "")


def latch_step(state, event, now):
    """Pure double-tap latch. No pending gap and no deferred release.

    state: phase idle | holding | latched, plus down_at, candidate,
    prev_up_at, prev_press_s.
    event: "down" | "up".
    Returns (new_state, action) with action in
    None | "start" | "release" | "latch" | "stop".

    Every release of a take that is not a short latch-candidate is "release"
    and is processed immediately. A short release only returns "latch" when
    this press began within _LATCH_GAP_S of a previous press shorter than
    _LATCH_MAX_PRESS_S. "stop" is the next down while latched. It clears
    prev_up_at and prev_press_s so that stop cannot count as a short release
    that arms the next tap. An up while latched does nothing.
    """
    st = dict(_LATCH_IDLE)
    st.update(state or {})
    phase = st.get("phase") or "idle"
    if event == "down":
        if phase == "latched":
            # A stop is not a short release. Leaving prev_up_at / prev_press_s
            # set made the next single tap inside the gap latch again.
            st.update(phase="idle", down_at=None, candidate=False,
                      prev_up_at=None, prev_press_s=None)
            return st, "stop"
        if phase == "holding":
            return st, None
        prev_up = st.get("prev_up_at")
        prev_press = st.get("prev_press_s")
        candidate = (
            prev_up is not None and prev_press is not None
            and (now - prev_up) < _LATCH_GAP_S
            and prev_press < _LATCH_MAX_PRESS_S)
        st.update(phase="holding", down_at=now, candidate=candidate)
        return st, "start"
    if event == "up":
        if phase == "latched":
            return st, None
        if phase != "holding":
            return st, None
        down_at = st.get("down_at")
        held = 0.0 if down_at is None else max(0.0, now - down_at)
        if st.get("candidate") and held < _LATCH_MAX_PRESS_S:
            st.update(phase="latched", candidate=False)
            return st, "latch"
        st.update(phase="idle", down_at=None, candidate=False,
                  prev_up_at=now, prev_press_s=held)
        return st, "release"
    return st, None


def modifiers_are_up():
    """True when ctrl, alt and shift are all up. Unknown OS state counts as up."""
    for vk in _MODIFIER_VKS:
        if _os_key_down(vk):
            return False
    return True


def wait_modifiers_up(timeout_s=_MODIFIER_WAIT_S):
    """Wait until ctrl, alt and shift are physically up. False on timeout."""
    deadline = time.monotonic() + float(timeout_s)
    while True:
        if modifiers_are_up():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def capture_selection():
    """Copy the current selection. Returns (text, saved, problem).

    problem is "modifiers" when ctrl, alt or shift stayed down, "nosel" when
    nothing was copied, and None when `text` is the selection.

    A unique sentinel is placed on the clipboard before Ctrl+C. If it is
    still there afterwards, the app copied nothing — including the case
    where a real selection would have equalled the previous clipboard.

    VS Code copies the current line, including its trailing newline, when
    nothing is selected. A real one-line selection has the same clipboard
    shape, so copy-line cannot be told apart and is a known limitation:
    both are accepted. The same limitation applies to any other app that
    copies the current line.

    On success, `capture_selection.context` is the foreground HWND, the
    focused-control HWND (GetGUIThreadInfo hwndFocus), and the
    GetLastInputInfo tick taken immediately after this Ctrl+C. Flow's own
    injection is therefore inside the stamp. A later user tick means the
    selection may have moved.

    `capture_selection.generation` is the clipboard generation right after
    this capture's own write (the sentinel). Restores of `saved` must use
    that value. A later cleanup must not read a new generation.
    """
    capture_selection.context = None
    capture_selection.generation = None
    try:
        saved = pyperclip.paste()
    except Exception:
        saved = ""
    # Before any write. A modifier timeout never copies, so this is the
    # generation a restore of `saved` is allowed to replace.
    capture_selection.generation = _clipboard_generation()
    if not wait_modifiers_up():
        print("[rewrite] modifiers still down; not sending ctrl+c")
        return None, saved, "modifiers"
    sentinel = f"flow-sel-{time.monotonic_ns()}-{os.getpid()}"
    if not _copy_clipboard(sentinel):
        return None, saved, "nosel"
    # Right after our own write. The app's Ctrl+C is not a Flow write.
    capture_selection.generation = _clipboard_generation()
    try:
        keyboard.send("ctrl+c")
    except Exception as e:
        print(f"[rewrite] could not copy the selection: {e}")
        return None, saved, "nosel"
    # Stamp after the injection. Checking GetLastInputInfo before Ctrl+C
    # would treat this send as user input and refuse every paste.
    capture_selection.context = {
        "hwnd": int(foreground_hwnd() or 0),
        "focus": int(focused_control_hwnd() or 0),
        "tick": _last_input_tick(),
    }
    time.sleep(0.08)
    try:
        current = pyperclip.paste()
    except Exception:
        current = ""
    if current == sentinel or not (current or "").strip():
        return None, saved, "nosel"
    return current, saved, None


def _last_input_tick():
    """GetLastInputInfo tick, or None where the OS call is unavailable.

    Tests monkeypatch this. The tick advances on key and mouse input,
    including keys Flow itself injects, so callers stamp it after those.
    """
    try:
        import ctypes
        from ctypes import wintypes
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]
        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        return int(info.dwTime)
    except Exception:
        return None


def _tick_newer(now, then):
    """True when `now` is a later GetLastInputInfo tick than `then`.

    The tick is a 32-bit millisecond count and wraps. A forward gap of
    under ~24 days counts as newer. Equal ticks are not newer.
    """
    if now is None or then is None:
        return False
    delta = (int(now) - int(then)) & 0xFFFFFFFF
    return 0 < delta < 0x80000000


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


# Each successful Flow clipboard write bumps this. A delayed restore records
# the value at schedule time and writes only if it is still current. A later
# job that keeps its result, or any newer Flow write, leaves the old restore
# with nothing to do.
_clip_gen = 0
_clip_gen_lock = threading.Lock()


def _clipboard_generation():
    with _clip_gen_lock:
        return _clip_gen


def _copy_clipboard(text, expect_gen=None):
    """Copy `text`, retrying on any failure. Never raises.

    Returns the clipboard generation after this write, taken under the same
    lock as the copy. A restore of this write passes that value back.
    `expect_gen` is a delayed restore's generation. None skips the check.
    A newer write returns None and does not copy. False is a failed copy.
    Callers that only need success can use the result as a boolean.
    """
    global _clip_gen
    last = None
    for delay in (0.0,) + _CLIP_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            with _clip_gen_lock:
                if expect_gen is not None and _clip_gen != expect_gen:
                    return None
                pyperclip.copy(text)
                _clip_gen += 1
                written = _clip_gen
            return written
        except Exception as e:
            last = e
    print(f"[paste] clipboard copy failed after "
          f"{1 + len(_CLIP_RETRY_DELAYS)} tries: {last}")
    return False


def paste_text(text, mode, restore=True):
    """Paste `text`. Returns the clipboard generation this write produced.

    The generation is the value `_copy_clipboard` captured in the same lock
    as the copy. Callers that restore afterwards pass it back. A typed paste
    and a failed copy return None.
    """
    print(f"[paste] mode={mode} target window: {foreground_window()!r}")
    if mode == "type":
        keyboard.write(text)
        return None
    old = None
    if restore:
        try:
            old = pyperclip.paste()
        except Exception:
            pass
    written = _copy_clipboard(text)
    if not written:
        # The transcript still has to land at the cursor. Typing is slower
        # but it does not depend on the clipboard. keyboard.write turns a
        # newline into Enter, which would send the message.
        print("[paste] clipboard unavailable; typing the dictation instead")
        keyboard.write(text.replace("\r\n", " ").replace("\n", " ").replace("\r", " "))
        return None
    time.sleep(0.02)
    keyboard.press("ctrl")
    keyboard.press("v")
    time.sleep(0.03)
    keyboard.release("v")
    keyboard.release("ctrl")
    print("[paste] ctrl+v sent")
    if old is not None:
        gen = written if type(written) is int else _clipboard_generation()

        def restore(gen=gen):
            time.sleep(0.6)
            wrote = _copy_clipboard(old, expect_gen=gen)
            if wrote is False:
                print("[paste] could not restore the previous clipboard")
        threading.Thread(target=restore, daemon=True).start()
    return written if type(written) is int else None


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
        self.summary_take = False  # shift pressed during this dictation take
        self.rewrite_take = False  # alt during this dictation: rewrite selection
        self.latched = False       # hands-free: key-up does not end the take
        self._latch = dict(_LATCH_IDLE)
        self._latch_owner = None   # "local" while a local take owns the gesture
        self._latch_pending = None # committed with recording=True, under the lock
        self._take_hwnd = 0        # foreground window when the take opened
        self._take_focus = 0       # focused control in that window at press
        self._paste_target = None  # hwnd/focus the open job pastes into
        self._paste_on_current = None
        self._input_tick = None    # GetLastInputInfo after our own injections
        self._latched_shown = None
        from collections import deque
        self._ptt_times = deque()
        self.hermes_context = None
        self.streamer = None
        self._pressed_keys = set()
        self._active_ptt_key = None
        self._take_key = None  # key that owns the open take; its up may end it
        self._take_gen = 0     # generation of that claim; a later press bumps it
        self._owner_token = None  # (key, gen) of the current claim
        # Owner key whose up is queued and whose claim is not cleared yet.
        # An ordinary take treats that as released. A latch does not.
        # The marker and the claim change together under _key_lock. Clearing
        # the marker first rejects the next key in that window.
        self._owner_up_queued = None
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
        _ov_on, _ov_hints = _overlay_flags(self.cfg)
        self.overlay = (Overlay(level_fn=lambda: self.recorder.level,
                                hints=_ov_hints,
                                hint_line=overlay_hint_line(self.cfg))
                        if _ov_on else None)
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
            # pystray can raise transient Win32 errors (WinError 1402, invalid
            # cursor handle). A tray glitch must never fail the take or leave
            # the overlay counting forever.
            try:
                self.tray.icon = make_icon(state)
            except Exception as e:
                print(f"[tray] icon update failed: {e}")
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
                    released_at=None, summary=False, rewrite=False,
                    full_samples=None, discarded=0, committed_at_release=None,
                    take_hwnd=0, input_tick=None, rewrite_capture=None,
                    release_hwnd=None, take_focus=0):
        """Hand a finished take to the single worker. Press order is paste order.

        A rewrite does not capture here. This runs on the ctrl thread, so it
        only reads the foreground HWND. Capture waits on modifiers and the
        clipboard, and process() does that on the job worker, one clipboard
        transaction at a time. A window that is no longer in front cancels
        the rewrite.
        """
        if mode == "hermes":
            self._job_q.put(("hermes", (audio, hermes_context)))
        elif mode == "command":
            self._job_q.put(("command", (audio,)))
        else:
            if rewrite and not _rewrite_enabled(self.cfg):
                rewrite = False
                rewrite_capture = None
            if rewrite and release_hwnd is None:
                release_hwnd = int(foreground_hwnd() or 0)
            hwnd, focus = unpack_take_target(take_hwnd, take_focus)
            # Pair, not a new trailing argument. release_hwnd stays last so
            # a queued rewrite can still read it from the end of the tuple.
            self._job_q.put(("dictate", (
                audio, streamer, released_at, summary, rewrite,
                full_samples, discarded, committed_at_release, (hwnd, focus),
                input_tick, rewrite_capture, release_hwnd)))

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

    def _claim_owner(self, key, gen=None):
        """Ctrl thread: this press owns the open take. Returns its generation.

        A later press claims a new generation. Failure cleanup for this press
        clears only when both the key and the generation still match.
        """
        lock = getattr(self, "_key_lock", None)

        def _do():
            allocated = gen
            if allocated is None:
                allocated = int(getattr(self, "_take_gen", 0) or 0) + 1
            allocated = int(allocated)
            self._take_gen = allocated
            self._take_key = key
            self._owner_token = (key, allocated)
            return allocated

        if lock is None:
            return _do()
        with lock:
            return _do()

    def _clear_take_owner_locked(self, key, gen):
        """Drop this claim and its release marker. Caller holds `_key_lock`.

        A newer generation is left in place, marker included.
        """
        if getattr(self, "_take_key", None) != key:
            return False
        if getattr(self, "_take_gen", None) != gen:
            return False
        self._take_key = None
        if getattr(self, "_owner_token", None) == (key, gen):
            self._owner_token = None
        if getattr(self, "_owner_up_queued", None) == key:
            self._owner_up_queued = None
        return True

    def _clear_take_owner(self, key, gen):
        """Drop this press's claim. A newer generation is left in place."""
        if key is None or gen is None:
            return False
        lock = getattr(self, "_key_lock", None)
        if lock is None:
            return self._clear_take_owner_locked(key, gen)
        with lock:
            return self._clear_take_owner_locked(key, gen)

    def _clear_owner_on_release(self, key):
        """End ownership for this release without wiping a later press.

        Callers that do not know the key (watchdog, direct on_release) clear
        the open take. A keyed release clears only the claim it still owns.
        The release marker stays until that claim is cleared. Both change
        under `_key_lock`, so a down in this window still sees the marker.
        """
        lock = getattr(self, "_key_lock", None)

        def _do():
            if key is None:
                self._take_key = None
                self._owner_token = None
                self._owner_up_queued = None
                return
            token = getattr(self, "_owner_token", None)
            if token is None:
                if getattr(self, "_take_key", None) == key:
                    self._take_key = None
                    if getattr(self, "_owner_up_queued", None) == key:
                        self._owner_up_queued = None
                return
            if token[0] == key:
                self._clear_take_owner_locked(token[0], token[1])

        if lock is None:
            _do()
        else:
            with lock:
                _do()

    def on_press(self, mode="dictate", key=None):
        stop_speech()
        # A second PTT key during a recording must return immediately. The
        # lock is held for the whole take, so waiting out _MIC_WAIT_S would
        # block the ctrl loop and then claim the microphone is being checked.
        with self._session_lock:
            if self.recording:
                return
        # Provisional ownership for this press, before the mic wait. Accepted
        # takes keep it. A failed start clears it only while the generation
        # still belongs to this press.
        claimed = self._claim_owner(key) if key is not None else None
        # Wait outside _session_lock: a probe may take seconds, and the
        # previous release still needs that lock. `busy` is processing, not
        # the mic — a new press records while the previous take decodes.
        if not self._mic_lock.acquire(timeout=_MIC_WAIT_S):
            with self._session_lock:
                recording = self.recording
            if not recording:
                self._note_mic_wait()
            self._clear_take_owner(key, claimed)
            return
        started = False
        opened = False
        try:
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
                    self.summary_take = False
                    self.rewrite_take = False
                    self.latched = False
                    self._latched_shown = None
                    self._take_hwnd = foreground_hwnd()
                    self._take_focus = focused_control_hwnd()
                    pending = getattr(self, "_latch_pending", None)
                    self._latch_pending = None
                    if (mode == "dictate" and pending
                            and self.cfg.get("mode") != "client"):
                        # Same locked section as recording=True. A remote clear
                        # that ran before this acquire cannot wipe it afterwards
                        # without observing owner == "local".
                        self._latch = pending
                        self._latch_owner = "local"
                        self.latched = pending.get("phase") == "latched"
                    self.recording = True
                    self._record_started_at = time.monotonic()
                    self._mic_held = True
                    started = True
                    if self.live_worker:
                        self.live_worker.pause("recorder-owner")
                    rec_gen = self.recorder.start()  # first — capture must not wait on mute/icon
                    opened = True
                    self.set_icon("rec", (f"Hermes · {context['title']}"
                                          if context else None))
                    if self.rewrite_take and self.overlay:
                        self.overlay.set(("rec", None, "rewrite"))
                    if self.cfg.get("mute_output_while_recording"):
                        self.muter.mute()
                    self._open_streamer(rec_gen)
            except Exception:
                if started and not opened:
                    with self._session_lock:
                        self.recording = False
                        self._record_started_at = None
                        self.latched = False
                    self._mic_held = False
                    started = False
                    self._clear_take_owner(key, claimed)
                raise
        finally:
            if not started:
                if not self.recording:
                    self._clear_take_owner(key, claimed)
                try:
                    self._mic_lock.release()
                except RuntimeError:
                    pass

    def _open_streamer(self, gen, handsfree=False):
        """Start the rolling decoder for the open take. Hands-free never types."""
        if self.active_mode != "dictate" or not self.cfg["streaming"]["enabled"]:
            self.streamer = None
            return
        if self.cfg.get("mode") == "client":
            if self.cfg["streaming"].get("client", True):
                self.streamer = ClientStreamer(self)
                self.streamer.start()
            return
        if handsfree or getattr(self, "latched", False):
            live = False
        else:
            # Feedback mode belongs to the skill, not to "does a style
            # exist" — see feedback_mode(). Defaults reproduce today's
            # behaviour exactly: matched -> ghost, unmatched -> live.
            live = (self.cfg["streaming"].get("live_type", True)
                    and feedback_mode(self.cfg, foreground_window()) == "live"
                    and not self.rewrite_take
                    and not self.summary_take)
        self.streamer = StreamWorker(self, live=live)
        self.streamer._gen = gen
        self.streamer.start()

    def _attach_frozen_raw(self, streamer, frozen):
        """Pair the sealed pre-edit transcript with the edited commit.

        `capture()` stores both under one lock. A 2-tuple stays a 2-tuple
        when there is no separate pre-edit text, so older callers still
        unpack `(text, samples)`.
        """
        if streamer is None or not isinstance(frozen, tuple) or len(frozen) < 2:
            return frozen
        if len(frozen) >= 3:
            return frozen
        raw = getattr(streamer, "_frozen_raw", None)
        if raw is None:
            raw = getattr(streamer, "committed_raw", None)
        if raw is None:
            return frozen
        text = frozen[0] or ""
        if raw == text or (raw == "" and text):
            return frozen
        return (frozen[0], frozen[1], raw)

    def _capture_take(self, streamer):
        """Audio plus the commit offsets, or stop() when the recorder is a fake."""
        cap = getattr(self.recorder, "capture", None)
        if cap is not None:
            try:
                audio, frozen = cap(streamer)
            except TypeError:
                pass
            else:
                return audio, self._attach_frozen_raw(streamer, frozen)
        audio = self.recorder.stop()
        if streamer is None:
            return audio, ("", 0)
        frozen = getattr(streamer, "_frozen", None)
        if frozen is None:
            text = getattr(streamer, "committed_text", "") or ""
            samples = int(getattr(streamer, "committed_samples", 0) or 0)
            frozen = (text, samples)
            raw = getattr(streamer, "committed_raw", None)
            if raw is not None:
                try:
                    streamer._frozen = frozen
                    streamer._frozen_raw = raw
                except Exception:
                    pass
        return audio, self._attach_frozen_raw(streamer, frozen)

    def _enable_summary(self):
        """Shift during a dictation take: stop live typing.

        Nothing typed yet: the summary is pasted at release. Words already
        on screen stay; the summary is copied instead and nothing is erased.
        """
        with self._session_lock:
            if not self.recording or self.active_mode != "dictate":
                return
            if getattr(self, "rewrite_take", False):
                return  # the first modifier wins
            self.summary_take = True
            if self.streamer is not None:
                self.streamer.live = False
        print("[summary] on for this take")
        if self.overlay:
            self.overlay.set(("rec", None, "summary"))

    def _enable_rewrite(self):
        """Alt down during a server-mode dictation that is already recording.

        Same shape as shift summary: the modifier arrives while the take is
        open. Live typing stops so it cannot replace the selection. Alt
        pressed first stays whatever that key did before rewrite existed
        (command mode when it is the command hotkey). Client mode has no
        rewrite. Disabled rewrite ignores the key. A press that arrives
        before recording is recording is ignored.
        """
        if not _rewrite_enabled(self.cfg):
            return
        if self.cfg.get("mode") == "client":
            print("[rewrite] ignored in client mode")
            return
        with self._session_lock:
            if not self.recording or self.active_mode != "dictate":
                return
            if getattr(self, "rewrite_take", False):
                return
            if getattr(self, "summary_take", False):
                return
            self.rewrite_take = True
            if self.streamer is not None:
                self.streamer.live = False
        print("[rewrite] on for this take")
        if self.overlay:
            self.overlay.set(("rec", None, "rewrite"))

    def _latched_words(self):
        streamer = getattr(self, "streamer", None)
        if streamer is None:
            return 0
        text = getattr(streamer, "committed_text", None)
        if text is None:
            text = getattr(streamer, "committed", "") or ""
        text = text or ""
        return len(text.split()) if text.strip() else 0

    def _latched_context_now(self):
        started = self._record_started_at
        elapsed = 0 if started is None else max(0.0, time.monotonic() - started)
        return format_latched_context(elapsed, self._latched_words())

    def _show_latched(self):
        self._latched_shown = self._latched_context_now()
        if self.overlay and not getattr(self, "summary_take", False) \
                and not getattr(self, "rewrite_take", False):
            self.overlay.set(("rec", self._latched_shown, "handsfree"))

    def on_release(self, key=None):
        """End the open take and queue it. Ordinary releases do this at once.

        The latch machine sets `_keep_latch` first. It has already stored the
        press length and release time the next candidate needs. Any other
        release disarms that memory so a command or a watchdog stop cannot
        become the first tap of a hands-free gesture.
        """
        keep_latch = bool(getattr(self, "_keep_latch", False))
        self._keep_latch = False
        current = None
        with self._session_lock:
            if not self.recording:
                self._clear_owner_on_release(key)
                return
            if self.active_mode == "remote":
                return  # remote session owns the recorder; the hotkey must not hijack it
            summary = getattr(self, "summary_take", False)
            rewrite = getattr(self, "rewrite_take", False)
            mode = self.active_mode
            hwnd = getattr(self, "_take_hwnd", 0) or 0
            focus = getattr(self, "_take_focus", 0) or 0
            self.recording = False
            self._record_started_at = None
            self._clear_owner_on_release(key)
            self.latched = False
            self._latched_shown = None
            self._latch_pending = None
            if not keep_latch:
                self._latch = dict(_LATCH_IDLE)
                self._latch_owner = None
            self.summary_take = False
            self.rewrite_take = False
            streamer = self.streamer
            self.streamer = None
            # Signal only. stop() joins the stream thread for up to 2s and
            # must not run here — this is the ctrl loop, and the next press
            # is waiting behind it. process() joins before it reads the commit.
            if streamer:
                streamer.halt()
            # Offsets and audio under one lock. A late trim cannot rebase
            # this pair or the next take.
            audio, committed = self._capture_take(streamer)
            full_samples = getattr(self.recorder, "last_full_samples", None)
            discarded = getattr(self.recorder, "last_discarded", 0)
            self.muter.restore()
            hermes_context = self.hermes_context
            self.hermes_context = None
            if self.live_worker and mode not in ("command", "hermes"):
                self.live_worker.resume("recorder-owner")
            # Baseline for "did the user type after we let go". Our own
            # injections during the hold are already in this tick.
            self._input_tick = _last_input_tick()
            input_tick = self._input_tick
            current = (mode, audio, streamer, summary, rewrite,
                       full_samples, discarded, committed, hermes_context,
                       hwnd, focus, input_tick)
        # The audio array is already detached. Free the mic before the
        # (possibly slow) decode so the next press can record immediately.
        self._release_mic()
        if current is not None:
            (mode, audio, streamer, summary, rewrite, full_samples,
             discarded, committed, hermes_context, hwnd, focus,
             input_tick) = current
            self._submit_job(
                mode, audio, streamer, hermes_context, time.monotonic(),
                summary=summary, rewrite=rewrite,
                full_samples=full_samples, discarded=discarded,
                committed_at_release=committed, take_hwnd=hwnd,
                take_focus=focus, input_tick=input_tick)

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
                             persist=True, full_samples=None, keep_audio=True):
        """Shared pipeline: resample -> ASR (tail only if streaming committed a
        prefix) -> cleanup -> store. Returns (text, meta).

        persist=False leaves the row in meta["pending"] so the dictate path
        can paste first and write history after.
        """
        target_sr = self.cfg["sample_rate"]
        if sample_rate and sample_rate != target_sr:
            audio = resample_linear(audio, sample_rate, target_sr)
        # A trimmed buffer is only the tail. Duration still describes the take.
        dur = ((full_samples if full_samples else len(audio)) / target_sr)
        if self.cfg.get("normalize_gain"):
            peak = float(np.abs(audio).max()) or 1.0
            audio = (audio / peak * 0.95).astype(np.float32)
        committed_text, committed_samples = committed[0], committed[1]
        # The third element is the pre-edit committed transcript, frozen at
        # release. Absent means the edited text is also the raw prefix.
        if len(committed) > 2 and committed[2] is not None:
            committed_raw = committed[2]
        else:
            committed_raw = committed_text
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
        # raw keeps the pre-edit committed transcript. `text` starts from
        # the edited prefix, so a streaming scratch is cleaned, not raw.
        raw = (committed_raw + " " + tail_raw).strip()
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
        # Trimmed audio is only the uncommitted tail, so it is not a valid
        # pair for the full transcript. Takes longer than the hold cap skip
        # the wav too — a hands-free hour must not write a giant file.
        try:
            cap_s = float(self.cfg.get("max_record_s", 600) or 600)
        except (TypeError, ValueError):
            cap_s = 600
        store_audio = audio
        if private or not keep_audio or dur > cap_s:
            store_audio = None
        if raw:
            meta["pending"] = {
                "raw": raw, "source": source, "window": window, "dur": dur,
                "asr_t": asr_t, "audio": store_audio,
                "sample_rate": target_sr, "cleaned": text, "private": private,
            }
        if persist:
            self._store_transcript(
                raw, source, window, dur, asr_t,
                store_audio, target_sr, text, private)
            meta.pop("pending", None)
        print(f"[{source}] [{dur:.1f}s audio -> {asr_t:.2f}s asr] {text}")
        return text, meta

    def _remember(self, text):
        """Keep the transcript before any paste attempt, so a failed paste
        can still be copied from the tray."""
        if text:
            self.last_text = text

    def _place_sticky(self, text, mode=None, restore=True, on_current=None):
        """Paste `text` into the target captured at press.

        Clipboard writes stay inside paste_text / _copy_clipboard. A quiet
        insert does not touch the clipboard. Rewrite does not call this:
        a rewrite is copied for the user to paste, and that path is unchanged.
        """
        if mode is None:
            mode = self.cfg.get("paste_mode", "clipboard")
        target = getattr(self, "_paste_target", None) or {"hwnd": 0, "focus": 0}
        bridge = getattr(self, "targets", None) or TargetBridge()

        def paste():
            return paste_text(text, mode, restore=restore)

        def copy():
            return _copy_clipboard(text)

        return sticky_deliver(
            text, target, bridge, paste, copy,
            enabled=sticky_enabled(self.cfg),
            on_current=on_current)

    def _safe_paste(self, text, mode, restore=True):
        """Paste, but never let a clipboard/typing error kill this thread.

        Returns the clipboard generation `paste_text` wrote, or None when
        nothing was written. A quiet insert and a copy-only refusal also
        return None: neither leaves a clipboard generation to restore.
        """
        if not text:
            return None
        try:
            if not sticky_enabled(self.cfg):
                return paste_text(text, mode, restore=restore)
            on_current = getattr(self, "_paste_on_current", None)
            result = self._place_sticky(
                text, mode, restore=restore, on_current=on_current)
            if result.get("action") == "copy":
                if self.overlay:
                    if result.get("copied"):
                        self.overlay.set(("say", result.get("message")
                                          or STICKY_UNAVAILABLE))
                    else:
                        self.overlay.set((
                            "say",
                            "Couldn't paste that dictation. "
                            "Tray menu: Copy last dictation."))
                return None
            gen = result.get("generation")
            return gen if type(gen) is int else None
        except Exception as e:
            print(f"[paste] failed, dictation kept in memory: {e}")
            if self.overlay:
                self.overlay.set((
                    "say",
                    "Couldn't paste that dictation. "
                    "Tray menu: Copy last dictation."))
            return None

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

    def _restore_clip(self, saved, delay=0.0, expect_gen=None):
        """Put the user's clipboard back if this job still owns it.

        `expect_gen` is the generation recorded at capture, right after that
        capture's own writes. Immediate and delayed restores both use it.
        Nothing here reads the generation again: a newer write must win.
        """
        if saved is None or expect_gen is None:
            return
        if delay and delay > 0:
            def go(gen=expect_gen):
                time.sleep(delay)
                _copy_clipboard(saved, expect_gen=gen)
            threading.Thread(target=go, daemon=True).start()
            return
        _copy_clipboard(saved, expect_gen=expect_gen)

    def _copy_kept_result(self, text, retain):
        """Put `text` on the clipboard. `retain` is this job's list, or None.

        Appending marks the job so process() does not restore a streamer's
        previous clipboard over the result. The list is local to the job.
        """
        if not _copy_clipboard(text):
            return False
        if retain is not None:
            retain.append(text)
        return True

    def _rewrite_captured_target(self):
        """Identity recorded by the capture that just ran, or a fresh read.

        A mocked capture_selection has no context. Production capture stamps
        hwnd, focus and the input tick inside capture_selection, immediately
        after its own Ctrl+C.
        """
        ctx = getattr(capture_selection, "context", None)
        if isinstance(ctx, dict):
            return ctx
        return {
            "hwnd": int(foreground_hwnd() or 0),
            "focus": int(focused_control_hwnd() or 0),
            "tick": _last_input_tick(),
        }

    def _bind_rewrite_target(self):
        """Capture the rewrite target before transcription.

        Disabled rewrite does not read or write the clipboard.
        Returns (target, abort_message). Modifiers must be up within 1.5s.
        If they are not, the rewrite is aborted with the existing message
        and nothing is copied from the selection. `target` holds the text,
        the saved clipboard, the clipboard generation after this capture's
        own write, and the hwnd, focus and input stamp taken after Flow's
        own Ctrl+C.
        """
        if not _rewrite_enabled(self.cfg):
            return None, None
        capture_selection.generation = None
        got = capture_selection()
        gen = getattr(capture_selection, "generation", None)
        if len(got) == 2:
            selection, saved = got
            problem = None if selection else "nosel"
        else:
            selection, saved, problem = got
        if problem == "modifiers":
            msg = "Let go of Ctrl, Alt and Shift — rewrite cancelled"
            if self.overlay:
                self.overlay.set(("say", msg))
            self._restore_clip(saved, expect_gen=gen)
            return None, msg
        if not selection:
            msg = "Select text first"
            if self.overlay:
                self.overlay.set(("say", msg))
            self._restore_clip(saved, expect_gen=gen)
            return None, msg
        captured = dict(self._rewrite_captured_target())
        captured["text"] = selection
        captured["saved"] = saved
        captured["gen"] = gen
        return captured, None

    def _bind_supplied_rewrite(self, rewrite_capture):
        """Saved clipboard and generation from a capture already in hand.

        No clipboard access and no waits. process() binds this before any
        early return so a short take, a deaf mic, or an empty transcript
        still puts the user's clipboard back.
        """
        target = None
        abort = None
        if isinstance(rewrite_capture, tuple) and len(rewrite_capture) == 2:
            target, abort = rewrite_capture
        saved = None
        gen = None
        if isinstance(target, dict) and not abort:
            saved = target.get("saved")
            gen = target.get("gen")
        return target, abort, saved, gen

    def _rewrite_selection(self, instruction, take_hwnd=0, retain=None,
                           target=None):
        """Leave a rewrite on the clipboard. Never inserts it.

        Disabled rewrite returns before any clipboard access.
        The selection is the capture in `target`, taken on the job worker
        before transcription. Direct callers capture here. Success leaves
        the rewritten text on the clipboard and the
        overlay tells the user to press Ctrl+V. There is no ctrl+v and no
        delayed restore. The user pastes, so `take_hwnd` is not checked.

        No selection, an LLM failure, a modifier timeout, or any exception
        after that capture puts the user's clipboard back and shows the
        existing message. `retain`, when passed, collects the copied text
        so this job does not restore a streamer's clipboard over it.
        """
        if not _rewrite_enabled(self.cfg):
            return None
        saved = None
        saved_gen = None
        restore = False
        try:
            if target is None:
                target, abort = self._bind_rewrite_target()
                if abort or not target:
                    return None
            selection = target.get("text")
            saved = target.get("saved")
            saved_gen = target.get("gen")
            restore = True
            if not selection:
                if self.overlay:
                    self.overlay.set(("say", "Select text first"))
                return None
            if self.overlay:
                self.overlay.set(("work", "Rewriting selection…"))
            try:
                rewritten = llm_rewrite(instruction, selection, self.cfg)
            except Exception:
                rewritten = None
                if self.overlay:
                    self.overlay.set((
                        "say", "Rewrite failed — selection left unchanged"))
                raise
            if not rewritten:
                if self.overlay:
                    self.overlay.set((
                        "say", "Rewrite failed — selection left unchanged"))
                return None
            if not wait_modifiers_up():
                if self.overlay:
                    self.overlay.set((
                        "say",
                        "Let go of Ctrl, Alt and Shift — rewrite cancelled"))
                return None
            if not self._copy_kept_result(rewritten, retain):
                if self.overlay:
                    self.overlay.set((
                        "say", "Rewrite failed — selection left unchanged"))
                return None
            restore = False
            print("[rewrite] copied; user pastes with ctrl+v")
            if self.overlay:
                self.overlay.set((
                    "say", "Rewritten — press Ctrl+V to replace"))
            return rewritten
        finally:
            if restore:
                self._restore_clip(saved, expect_gen=saved_gen)

    def _apply_voice_edit(self, text, live_typed):
        """Scratch-that on text that is not already on screen.

        Already-typed sentences stay. The phrase itself is dropped from the
        untyped tail. Nothing is backspaced.
        """
        return voice_edit_keeping_screen(text or "", live_typed, self.cfg)

    def _consume_paste_last(self, text, live_typed):
        """Replace a whole-take paste-last command with the previous dictation.

        Returns (text, live_typed, phrase). `phrase` is None when this take
        is ordinary text. A command does not write the phrase into last_text.
        An empty previous dictation shows the overlay and pastes nothing.
        """
        decision = apply_paste_last(
            text or "", live_typed, getattr(self, "last_text", "") or "",
            self.cfg)
        if decision["kind"] != "command":
            return text, live_typed, None
        if decision["empty"]:
            print("[voice] paste last, nothing saved")
            if self.overlay:
                self.overlay.set(("say", PASTE_LAST_EMPTY))
            return "", 0, decision["history"]
        print("[voice] paste last")
        return decision["paste"], 0, decision["history"]

    def _summary_target_ok(self, take_hwnd, input_tick):
        """Whether a summary of an untyped take may be pasted.

        Returns "paste", or ("copy", message). A changed window, a modifier
        still held, or any user input since the release means the summary
        stays on the clipboard and nothing is pasted. This never erases.
        Live-typed takes do not get here: those copy the summary and leave
        the dictated text on screen.
        """
        if not wait_modifiers_up():
            print("[summary] modifiers still down; copying the summary")
            return ("copy", "Summary copied")
        now = foreground_hwnd()
        if take_hwnd and now and now != take_hwnd:
            print("[summary] window changed; copying the summary")
            return ("copy", "Summary copied — window changed")
        if _tick_newer(_last_input_tick(), input_tick):
            print("[summary] user input since release; copying the summary")
            return ("copy", "Summary copied")
        return "paste"

    def process(self, audio, streamer=None, released_at=None, summary=False,
                rewrite=False, full_samples=None, discarded=0,
                committed_at_release=None, take_hwnd=0, input_tick=None,
                rewrite_capture=None, release_hwnd=None):
        take_hwnd, take_focus = unpack_take_target(take_hwnd)
        self._paste_target = {"hwnd": take_hwnd, "focus": take_focus}
        self._paste_on_current = None
        stop = threading.Event()
        started = time.monotonic() if released_at is None else released_at
        watcher = threading.Thread(
            target=self._announce_wait, args=(started, stop),
            daemon=True, name="flow-wait")
        watcher.start()
        with self.busy:
            try:
                # This job only. Cleared by falling out of process(); nothing
                # on self remembers a clipboard decision for the next take.
                retain_clip = False
                # Clipboard saved by a rewrite capture, and the generation
                # recorded right after that capture's own write. The finally
                # restores only while that generation is still current.
                # Bound before any early return when the capture was supplied.
                rewrite_saved = None
                rewrite_gen = None
                rewrite_target = None
                rewrite_abort = None
                # Generation the final paste wrote, captured with that write.
                # Cleanup of a streamed take uses this, not an older clip_gen.
                paste_gen = None
                if rewrite and not _rewrite_enabled(self.cfg):
                    # No capture, no sentinel write, no rewrite cleanup.
                    rewrite = False
                    rewrite_capture = None
                if rewrite and rewrite_capture is not None:
                    (rewrite_target, rewrite_abort,
                     rewrite_saved, rewrite_gen) = self._bind_supplied_rewrite(
                         rewrite_capture)

                def _note_paste(payload, mode, restore):
                    nonlocal paste_gen
                    wrote = self._safe_paste(payload, mode, restore=restore)
                    if type(wrote) is int:
                        paste_gen = wrote
                    return wrote

                # Inside the try: whatever fails from here on, the finally
                # stops the "Transcribing… (Ns)" ticker.
                self.set_icon("work")
                work_started = time.monotonic()
                queue_s = (0.0 if released_at is None
                           else max(0.0, work_started - released_at))
                samples = full_samples if full_samples else len(audio)
                dur = samples / self.cfg["sample_rate"]
                if dur < self.cfg["min_duration_s"]:
                    return
                # A trimmed buffer is the tail. Silence there is a pause at
                # the end of a real take, not a dead microphone.
                if not discarded and self.mic_is_deaf(audio):
                    return
                if self.cfg["mode"] == "client":
                    if isinstance(streamer, ClientStreamer) and streamer.sid:
                        text, typed = streamer.finish()
                        if text is None:  # server lost the session mid-flight
                            text, typed = self.client_transcribe(audio, "/transcribe"), 0
                        text, typed, phrase = self._consume_paste_last(
                            text, typed)
                        if phrase is not None:
                            if text:
                                _note_paste(text, "clipboard", True)
                            return
                        self._remember(text)
                        remainder = (text or "")[typed:]
                        if remainder:
                            _note_paste(remainder, "clipboard", typed == 0)
                        if typed:
                            time.sleep(0.3)
                            gen = (paste_gen if type(paste_gen) is int
                                   else getattr(streamer, "clip_gen", None))
                            try:
                                if gen is None:
                                    streamer.restore_clipboard()
                                else:
                                    streamer.restore_clipboard(gen)
                            except TypeError:
                                streamer.restore_clipboard()
                    else:
                        text = self.client_transcribe(audio, "/transcribe")
                        text, _typed, phrase = self._consume_paste_last(text, 0)
                        if phrase is not None:
                            if text:
                                _note_paste(text, self.cfg["paste_mode"], True)
                            return
                        self._remember(text)
                        _note_paste(text, self.cfg["paste_mode"], True)
                    return
                join_s = 0.0
                if streamer:
                    # Join only after the rolling pass has been halted, so
                    # committed_text cannot move under the final decode.
                    # The offsets used below were captured with the audio.
                    j0 = time.monotonic()
                    streamer.stop()
                    join_s = time.monotonic() - j0
                    live_typed = streamer.typed_len
                else:
                    live_typed = 0
                if committed_at_release is not None:
                    committed = committed_at_release
                elif streamer is not None:
                    committed = self._attach_frozen_raw(
                        streamer,
                        (getattr(streamer, "committed_text", "") or "",
                         int(getattr(streamer, "committed_samples", 0) or 0)))
                else:
                    committed = ("", 0)
                # Capture on this worker, in line with every other clipboard
                # copy and paste. The ctrl thread only stored the foreground
                # HWND at release. A different window cancels the rewrite
                # and this job does not touch the clipboard. A capture that
                # was already supplied is bound above, before any return.
                if rewrite and rewrite_capture is None:
                    now = int(foreground_hwnd() or 0)
                    released = int(release_hwnd or 0)
                    if released and now and released != now:
                        rewrite_abort = (
                            "Selection changed — rewrite cancelled")
                    else:
                        rewrite_capture = self._bind_rewrite_target()
                        (rewrite_target, rewrite_abort,
                         rewrite_saved, rewrite_gen) = (
                             self._bind_supplied_rewrite(rewrite_capture))
                text, meta = self.transcribe_and_store(
                    audio, source="pc-hotkey", window=foreground_window(),
                    committed=committed, skip_cleanup=live_typed > 0,
                    persist=False, full_samples=full_samples,
                    keep_audio=not discarded)
                text, live_typed = self._apply_voice_edit(text, live_typed)
                # Voice edits run after the row is built. A paste-last command
                # stores the spoken phrase and pastes the previous dictation.
                # last_text is not replaced by the phrase.
                paste_last_phrase = None
                if not rewrite and not summary and text:
                    text, live_typed, paste_last_phrase = self._consume_paste_last(
                        text, live_typed)
                _pending = (meta or {}).get("pending") if meta else None
                if isinstance(_pending, dict):
                    if paste_last_phrase is not None:
                        _pending["raw"] = paste_last_phrase
                        _pending["cleaned"] = paste_last_phrase
                        _pending["source"] = "pc-hotkey"
                    else:
                        _pending["cleaned"] = text
                if rewrite:
                    if rewrite_abort or not rewrite_target:
                        if rewrite_abort and self.overlay:
                            self.overlay.set(("say", rewrite_abort))
                        self._remember("")
                        text = ""
                        live_typed = 0
                    elif not text:
                        self._remember("")
                        live_typed = 0
                    else:
                        kept = []
                        # _rewrite_selection restores on its own failure.
                        rewrite_saved = None
                        out = self._rewrite_selection(
                            text, take_hwnd, kept, rewrite_target)
                        retain_clip = bool(kept)
                        self._remember(out or "")
                        pending = (meta or {}).get("pending")
                        if pending and out:
                            pending["cleaned"] = out
                        # The rewrite is on the clipboard, or nothing was
                        # selected. Don't also paste the spoken instruction.
                        text = ""
                        live_typed = 0
                elif summary and text:
                    if self.overlay:
                        self.overlay.set(("work", "Making it to the point…"))
                    short = llm_summarize(text, self.cfg)
                    if short:
                        pending = (meta or {}).get("pending")
                        if pending:
                            pending["cleaned"] = short
                        if live_typed:
                            # Already on screen. Never backspace it. The
                            # summary waits on the clipboard. A failed copy
                            # stays in last_text; the line must not say paste.
                            if self._copy_kept_result(short, None):
                                retain_clip = True
                                message = "Summary copied — paste to replace"
                            else:
                                message = ("Summary ready — use Copy "
                                           "last dictation")
                            self._remember(short)
                            if self.overlay:
                                self.overlay.set(("say", message))
                            text = ""
                            live_typed = 0
                        else:
                            decision = self._summary_target_ok(
                                take_hwnd, input_tick)
                            if decision != "paste":
                                _kind, message = decision
                                # The window moved. Try the original box
                                # before giving up. Modifiers and a newer
                                # input tick still refuse, inside on_current,
                                # if the user has come back to that window.
                                if (message == "Summary copied — window changed"
                                        and sticky_enabled(self.cfg)):
                                    # Remember before the paste. A focus or
                                    # clipboard error must not drop the summary.
                                    self._remember(short)
                                    try:
                                        result = self._place_sticky(
                                            short,
                                            on_current=lambda: (
                                                self._summary_target_ok(
                                                    take_hwnd, input_tick)))
                                    except Exception as e:
                                        print(f"[summary] sticky paste failed: {e}")
                                        result = {"pasted": False, "copied": False}
                                    if result.get("pasted"):
                                        gen = result.get("generation")
                                        if type(gen) is int:
                                            paste_gen = gen
                                    else:
                                        if result.get("copied"):
                                            retain_clip = True
                                            message = (result.get("message")
                                                       or STICKY_UNAVAILABLE)
                                        else:
                                            message = ("Summary ready — use Copy "
                                                       "last dictation")
                                        if self.overlay:
                                            self.overlay.set(("say", message))
                                    text = ""
                                    live_typed = 0
                                else:
                                    if self._copy_kept_result(short, None):
                                        retain_clip = True
                                    else:
                                        message = ("Summary ready — use Copy "
                                                   "last dictation")
                                    self._remember(short)
                                    if self.overlay:
                                        self.overlay.set(("say", message))
                                    text = ""
                                    live_typed = 0
                            else:
                                text = short
                                self._remember(short)
                                self._paste_on_current = (
                                    lambda: self._summary_target_ok(
                                        take_hwnd, input_tick))
                    elif self.overlay:
                        self.overlay.set(("say", "Summary failed — pasted what you said"))
                if not rewrite and not (summary and not text and retain_clip):
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
                            _note_paste(remainder, "clipboard", False)
                        elif text:
                            _note_paste(apply_replacements(text, self.cfg),
                                        self.cfg["paste_mode"], True)
                    except Exception as e:
                        print(f"[paste] replacements failed, pasting the "
                              f"transcript unchanged: {e}")
                        if text and live_typed:
                            _note_paste(text[live_typed:], "clipboard", False)
                        elif text:
                            _note_paste(text, self.cfg["paste_mode"], True)
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
                if rewrite_saved is not None:
                    self._restore_clip(rewrite_saved, expect_gen=rewrite_gen)
                    rewrite_saved = None
                if streamer:
                    # The worker recorded clip_gen at save and after its own
                    # writes. Do not read the generation here: a newer take's
                    # paste already owns the clipboard.
                    time.sleep(0.3)
                    if retain_clip:
                        # This job left its result on the clipboard. Drop the
                        # saved clip so nothing later on this streamer puts
                        # it back. retain_clip itself dies with the call.
                        streamer.saved_clip = None
                    else:
                        restore = getattr(streamer, "restore_clipboard", None)
                        if restore:
                            # The final paste's generation, taken with that
                            # write. Fall back to the streamer's own record
                            # when this job did not paste a tail.
                            gen = (paste_gen if type(paste_gen) is int
                                   else getattr(streamer, "clip_gen", None))
                            try:
                                if gen is None:
                                    restore()
                                else:
                                    restore(gen)
                            except TypeError:
                                restore()
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
                take_gen = self.recorder.start()
                # Nothing local is recording: this acquire proved it. Drop a
                # finished gesture so it cannot arm the next local tap, then
                # mark the session remote so a later clear is not a local one.
                self._latch_owner = None
                self._clear_latch()
                self._latch_owner = "remote"
                self.set_icon("rec")
                if self.cfg.get("mute_output_while_recording"):
                    self.muter.mute()
                if self.cfg["streaming"]["enabled"]:
                    # Host-side streaming for the VM bridge: commits accumulate
                    # while the key is held; the daemon polls /record/partial and
                    # types committed deltas live. voice_edits is off, same as
                    # StreamSession: the VM has already typed the published
                    # prefix, and a scratch edit would make its delta corrupt
                    # ("Drop me." + "s."). Local hands-free keeps edits.
                    self.remote_streamer = StreamWorker(
                        self, live=False, voice_edits=False)
                    self.remote_streamer._gen = take_gen
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
            if streamer:
                streamer.halt()
            audio, committed = self._capture_take(streamer)
            full_samples = getattr(self.recorder, "last_full_samples", None)
            discarded = getattr(self.recorder, "last_discarded", 0)
            self._clear_latch()
            self.muter.restore()
            if self.live_worker:
                self.live_worker.resume("recorder-owner")
        if streamer:
            streamer.stop()  # join off the lock; offsets were frozen in capture
        self._release_mic()
        with self.busy:
            self.set_icon("work")
            try:
                samples = full_samples if full_samples else len(audio)
                if samples / self.cfg["sample_rate"] < self.cfg["min_duration_s"]:
                    return ""
                partial_served = bool(streamer and streamer.committed_text)
                text, _ = self.transcribe_and_store(
                    audio, source="vm-bridge", window=window,
                    committed=committed, skip_cleanup=partial_served,
                    full_samples=full_samples, keep_audio=not discarded)
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
                streamer.halt()
            self.recorder.stop()  # seals the generation before the next take
            self._clear_latch()
            self.muter.restore()
            if self.live_worker:
                self.live_worker.resume("recorder-owner")
            self.restore_base_state()
        if streamer:
            streamer.stop()
        self._release_mic()
        print("[remote] recording cancelled")
        return True

    def _clear_latch(self):
        """Drop latch state that is not a local take's.

        Remote stop and cancel call this while they hold `_session_lock`.
        A local take sets `_latch_owner` to "local" in the same locked
        section that sets `recording`. Clearing past that point used to
        erase the next take's phase, so its release did nothing and the
        mic stayed open.
        """
        if getattr(self, "_latch_owner", None) == "local":
            return
        self.latched = False
        self._latch = dict(_LATCH_IDLE)
        self._latch_pending = None
        self._latched_shown = None

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
                    # The synthesized up has not reached on_release yet.
                    # Publish the marker with the drop, or the next key is
                    # rejected while this owner is still claimed.
                    self._publish_owner_release_locked()
            self._pressed_keys.add(name)
            owner = getattr(self, "_take_key", None)
            # A key that does not own the open take cannot adopt it. Its up
            # then has nothing to release. Right alt during a latched summary,
            # or a customized command key during a latch, stays a rejected
            # secondary press. An ordinary take whose owner-up is already
            # queued has released: the next key is accepted, as at 9762188.
            if (self._active_ptt_key is None and owner not in (None, name)
                    and not self._ordinary_owner_release_queued(name)):
                pass
            elif self._active_ptt_key is None:
                self._active_ptt_key = name
                self._os_up_streak = 0
                # The key rides on the queue. The ctrl thread claims
                # _take_key when the take is accepted, with that press's
                # generation. A failed start clears only that claim.
                start = True
        if missed_up:
            print(f"[hotkey] {name!r} down {gap:.1f}s after the last event; "
                  "treating it as a missed release")
            self._stamp_ptt(now)
            self._ctrl_q.put(("up", mode, name))
        if start:
            self._stamp_ptt(now)
            self._ctrl_q.put(("down", mode, name))

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
                # Only the key that owns the take. A rejected secondary press
                # that managed to sit in _active_ptt_key must not end it.
                # The next key of an ordinary take whose owner-up is already
                # queued is released too, or its down would have no up.
                owner = getattr(self, "_take_key", None)
                if (owner in (None, name)
                        or self._ordinary_owner_release_queued(name)):
                    release = True
                    if owner == name and not getattr(self, "latched", False):
                        self._owner_up_queued = name
        if release:
            self._stamp_ptt(time.monotonic())
            self._ctrl_q.put(("up", mode, name))

    def _stamp_ptt(self, when):
        q = getattr(self, "_ptt_times", None)
        if q is None:
            from collections import deque
            q = deque()
            self._ptt_times = q
        q.append(when)

    def _pop_ptt_time(self):
        q = getattr(self, "_ptt_times", None)
        if not q:
            return time.monotonic()
        return q.popleft()

    def _on_key(self, event):
        # This runs inside the keyboard listener thread — it MUST be trivial and
        # never throw, or the listener dies and every hotkey stops working until
        # restart. So: match, drop auto-repeat, enqueue. All real work (mute,
        # audio start, transcription) happens in _ctrl_loop. The lock below is
        # uncontended and held only for the set update.
        try:
            name = event.name
            down = event.event_type == keyboard.KEY_DOWN
            # getattr: a half-built Flow (and any future caller) must not
            # throw out of the listener. A throw kills every hotkey.
            recording = getattr(self, "recording", False)
            active = getattr(self, "active_mode", None)
            # Alt during an open server-mode dictation is rewrite, the same
            # way shift is summary. Alt pressed first is not rewritten into
            # a chord: it stays the command hotkey when that is how it is
            # configured. Client mode has no rewrite. Disabled rewrite
            # leaves the key alone: no mode change and no clipboard.
            cfg = getattr(self, "cfg", None)
            if (down and name in _REWRITE_KEYS and recording
                    and active == "dictate"
                    and isinstance(cfg, dict)
                    and cfg.get("mode") != "client"
                    and _rewrite_enabled(cfg)
                    and not getattr(self, "rewrite_take", False)
                    and not getattr(self, "summary_take", False)):
                self._ctrl_q.put(("rewrite", None))
                return
            if (name in _SUMMARY_KEYS and down
                    and recording and active == "dictate"
                    and not getattr(self, "summary_take", False)
                    and not getattr(self, "rewrite_take", False)):
                self._ctrl_q.put(("summary", None))
                return
            if name == self.cfg.get("live_hotkey"):
                self._on_live_key(event, name)
                return
            mode = self._ptt_mode(name)
            if mode is None:
                return
            if down:
                self._on_ptt_down(name, mode)
            elif event.event_type == keyboard.KEY_UP:
                self._on_ptt_up(name, mode)
        except Exception as e:
            print(f"[hotkey] listener error: {e}")

    def _publish_owner_release_locked(self):
        """Note that the held key is being dropped. Caller holds `_key_lock`.

        The marker stays until ownership clears. A latched take is not an
        ordinary release: the watchdog forgets a stuck key without ending
        the take, and that disarm must not publish this marker.
        """
        if getattr(self, "latched", False):
            return
        owner = getattr(self, "_take_key", None)
        if owner:
            self._owner_up_queued = owner

    def _drop_held_key_locked(self):
        """Forget the PTT key we think is held. Caller holds `_key_lock`.

        The owner-release marker is published in this same hold, and it
        stays until ownership clears.
        """
        name = self._active_ptt_key
        if name:
            self._pressed_keys.discard(name)
            self._key_seen_at.pop(name, None)
        self._active_ptt_key = None
        self._os_up_streak = 0
        self._publish_owner_release_locked()
        return name

    def _drop_held_key(self):
        """Forget the PTT key we think is held. Caller runs the release path."""
        with self._key_lock:
            return self._drop_held_key_locked()

    def _poll_key_watchdog(self):
        """Synthesize KEY_UP when the OS says the held PTT key is up, twice.

        A release over an elevated window, the lock screen, or a hook stall
        never delivers KEY_UP. The next real press is then swallowed by the
        auto-repeat guard and the mic stays open. One poll is not enough.
        A latched take is supposed to have the key up — don't end it.
        A lost KEY_UP while latched still has to clear the pressed-key set,
        or the next tap inside a second is swallowed as auto-repeat.
        """
        if getattr(self, "latched", False):
            self._reconcile_latched_key()
            return
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
            name = self._drop_held_key_locked()
        print(f"[hotkey] {name!r} still marked held but the OS reports it up; "
              "releasing")
        self.on_release()

    def _reconcile_latched_key(self):
        """While latched, forget a key the OS says is up. Do not end the take."""
        with self._key_lock:
            name = self._active_ptt_key or (self.cfg.get("hotkey")
                                            if self.cfg else None)
            if not name:
                return
            if name not in self._pressed_keys and self._active_ptt_key is None:
                self._os_up_streak = 0
                return
            vk = _vk_for_hotkey(name)
            if vk is None:
                return
            down = _os_key_down(vk)
            if down is None or down:
                self._os_up_streak = 0
                return
            self._os_up_streak += 1
            if self._os_up_streak < _KEY_UP_POLLS:
                return
            self._pressed_keys.discard(name)
            self._key_seen_at.pop(name, None)
            if self._active_ptt_key == name:
                self._active_ptt_key = None
            self._os_up_streak = 0

    def _poll_record_cap(self):
        """Force-release a local recording that has run past its cap.

        A held key uses max_record_s. A hands-free latch uses max_latched_s
        (the key is up on purpose, so this is the only orphan guard).
        Remote sessions keep their own cancel timer, and that one discards.
        This path goes through on_release so the audio is transcribed.
        """
        if not self.recording or getattr(self, "active_mode", None) == "remote":
            return
        latched = getattr(self, "latched", False)
        key = "max_latched_s" if latched else "max_record_s"
        default = 7200 if latched else 600
        try:
            cap = float(self.cfg.get(key, default))
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

    def _latch_event(self, event, now, key=None):
        """Apply one dictate-hotkey event. Client mode never latches.

        The phase change is committed under `_session_lock` together with
        the recording flag (inside on_press / on_release / here). Local taps
        are ignored while a remote session owns the recorder. A latched
        phase with nothing recording is stale and must not swallow the next
        press. Remote cleanup cannot clear a phase this method has committed
        with owner "local".
        """
        if self.cfg.get("mode") == "client":
            if event == "down":
                if key is None:
                    self.on_press("dictate")
                else:
                    self.on_press("dictate", key=key)
            elif event == "up":
                if key is None:
                    self.on_release()
                else:
                    self.on_release(key)
            return
        with self._session_lock:
            if self.recording and getattr(self, "active_mode", None) == "remote":
                return
            state = dict(getattr(self, "_latch", None) or _LATCH_IDLE)
            if not self.recording and state.get("phase") == "latched":
                state = dict(_LATCH_IDLE)
                self._latch = state
                self.latched = False
                self._latch_owner = None
            new_state, action = latch_step(state, event, now)
            if action == "start":
                # Park it. on_press publishes the phase in the same section
                # that sets recording, after it wins the lock.
                self._latch_pending = new_state
                self._latch_owner = "local"
            elif action == "latch":
                self._latch = new_state
                self.latched = True
                self._latch_owner = "local"
            elif action in ("release", "stop"):
                self._latch = new_state
                self.latched = False
                self._latch_owner = "local"
            else:
                self._latch = new_state
                self.latched = new_state.get("phase") == "latched"
                if action is None and new_state.get("phase") == "latched":
                    self._latch_owner = "local"
        if action == "start":
            try:
                if key is None:
                    self.on_press("dictate")
                else:
                    self.on_press("dictate", key=key)
            finally:
                with self._session_lock:
                    if not self.recording and not self.latched:
                        # The press did not open a take. Do not leave a local
                        # owner behind, or a later remote clear would refuse
                        # to run. Key ownership is cleared inside on_press,
                        # and only for this press's generation.
                        self._latch_pending = None
                        self._latch_owner = None
                        st = dict(self._latch or _LATCH_IDLE)
                        if st.get("phase") == "holding":
                            st["phase"] = "idle"
                            st["candidate"] = False
                            self._latch = st
        elif action == "latch":
            self._become_latched()
        elif action in ("release", "stop"):
            self._keep_latch = True
            if key is None:
                self.on_release()
            else:
                self.on_release(key)

    def _become_latched(self):
        """The candidate release was short. Keep this recording, stop typing it."""
        with self._session_lock:
            if not self.recording or getattr(self, "active_mode", None) == "remote":
                return
            if self.cfg.get("mode") == "client":
                return
            self.latched = True
            self._latch_owner = "local"
            st = dict(self._latch or _LATCH_IDLE)
            st["phase"] = "latched"
            st["candidate"] = False
            self._latch = st
            if self.streamer is not None:
                self.streamer.live = False
        self._show_latched()

    def _ordinary_owner_release_queued(self, name):
        """True when an ordinary take's owner-up is queued and `name` is another key.

        The ctrl thread has not drained that up, so `_take_key` still names
        the owner. A latched take keeps its owner: the key is up on purpose.
        Caller holds `_key_lock`.
        """
        if getattr(self, "latched", False):
            return False
        owner = getattr(self, "_take_key", None)
        if owner in (None, name):
            return False
        return getattr(self, "_owner_up_queued", None) == owner

    def _handle_up(self, mode, when, key=None):
        """Key-up from the ctrl loop.

        Client mode is the pre-latch hotkey: down starts, up ends.
        The release marker stays set until on_release clears the claim.
        """
        if mode == "dictate" and self.cfg.get("mode") != "client":
            self._latch_event("up", when, key=key)
        elif key is None:
            self.on_release()
        else:
            self.on_release(key)

    def _poll_latched_clock(self):
        """Hands-free context line: elapsed and word count, about once a second."""
        if not getattr(self, "latched", False) or not self.recording:
            return
        if getattr(self, "summary_take", False) or getattr(self, "rewrite_take", False):
            return
        if self.overlay is None or self._record_started_at is None:
            return
        shown = self._latched_context_now()
        if shown == getattr(self, "_latched_shown", None):
            return
        self._latched_shown = shown
        self.overlay.set(("rec", shown, "handsfree"))

    def _ctrl_loop(self):
        import queue
        while not getattr(self, "_ctrl_stop", False):
            try:
                item = self._ctrl_q.get(timeout=_KEY_POLL_S)
            except queue.Empty:
                if getattr(self, "_ctrl_stop", False):
                    return
                try:
                    self._poll_key_watchdog()
                    self._poll_record_cap()
                    self._poll_latched_clock()
                except Exception as e:
                    # Don't clear `recording` here: a failed poll must not
                    # orphan the open mic. The next poll retries.
                    print(f"[hotkey] watchdog error: {e}")
                continue
            if getattr(self, "_ctrl_stop", False):
                return
            action = item[0]
            mode = item[1] if len(item) > 1 else None
            key = item[2] if len(item) > 2 else None
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
                elif action == "summary":
                    self._enable_summary()
                elif action == "rewrite":
                    self._enable_rewrite()
                elif action == "down":
                    when = self._pop_ptt_time()
                    if mode == "dictate" and self.cfg.get("mode") != "client":
                        self._latch_event("down", when, key=key)
                    elif key is None:
                        self.on_press(mode)
                    else:
                        self.on_press(mode, key=key)
                elif action == "up":
                    when = self._pop_ptt_time()
                    self._handle_up(mode, when, key=key)
                else:
                    self.on_release()
            except Exception as e:
                # Never let a press/release error propagate — recover the state.
                # Clear this press's owner only. A later queued press keeps its.
                print(f"[hotkey] {action} handler error: {e}")
                self.recording = False
                self._record_started_at = None
                if key is not None:
                    self._clear_owner_on_release(key)
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
