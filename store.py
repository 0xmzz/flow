"""Local transcript store + audio helpers for Flow.

Every dictation is appended to history.jsonl. Optionally the paired audio is
archived under dataset/<date>/<id>.wav, giving a fine-tuning-ready corpus of
your own voice that never leaves the machine.
"""

import io
import json
import threading
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def save_wav(path, audio, sample_rate):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def decode_wav(raw):
    """Decode WAV bytes -> (float32 mono in [-1,1], sample_rate)."""
    with wave.open(io.BytesIO(raw), "rb") as w:
        ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sw} bytes")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a.astype(np.float32), sr


def resample_linear(audio, sr_in, sr_out):
    if sr_in == sr_out:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * sr_out / sr_in))
    if n_out <= 1:
        return audio.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class TranscriptStore:
    def __init__(self, base_dir, archive_audio=False):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.history_path = self.base / "history.jsonl"
        self.audio_dir = self.base / "dataset"
        self.archive_audio = archive_audio
        self._lock = threading.Lock()

    def add(self, text, *, source, window=None, duration_s=None, asr_s=None,
            model=None, device=None, audio=None, sample_rate=16000):
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "text": text,
            "source": source,
            "window": window,
            "duration_s": round(duration_s, 2) if duration_s else None,
            "asr_s": round(asr_s, 2) if asr_s else None,
            "words": len(text.split()),
            "model": model,
            "device": device,
        }
        if self.archive_audio and audio is not None and len(audio):
            day = entry["ts"][:10]
            d = self.audio_dir / day
            d.mkdir(parents=True, exist_ok=True)
            wav = d / f"{entry['id']}.wav"
            save_wav(wav, audio, sample_rate)
            entry["audio"] = str(wav.relative_to(self.base)).replace("\\", "/")
        with self._lock, self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def read(self, limit=None):
        if not self.history_path.exists():
            return []
        with self._lock:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
        entries = [json.loads(l) for l in lines if l.strip()]
        return entries[-limit:] if limit else entries

    def stats(self):
        entries = self.read()
        words = sum(e.get("words", 0) for e in entries)
        by_source = {}
        for e in entries:
            by_source[e.get("source", "?")] = by_source.get(e.get("source", "?"), 0) + 1
        return {"dictations": len(entries), "words": words, "by_source": by_source}
