"""Flow fallback ASR node — runs on multivac's RTX 4080 (Laptop, 12GB).

Always-on, so phone/thin-client dictation survives beast being asleep, and
remote load stays off the actively-used 5090. Degraded on purpose: one-shot
transcription only (no streaming, no LLM cleanup) — dictation that always
works beats features that sometimes do.

Routes (Bearer token, same token as beast):
  GET  /health      -> {ok, model, node}
  POST /transcribe  -> wav/m4a/ogg body -> {text, asr_s, duration_s}
"""

import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = os.environ.get("ASR_MODEL", "large-v3")
PORT = int(os.environ.get("ASR_PORT", "8757"))
TOKEN = os.environ.get("ASR_TOKEN", "")

VOCAB = os.environ.get("ASR_VOCAB", "")  # comma-separated, mirrors beast vocab
VAD = {"threshold": 0.3, "min_silence_duration_ms": 400, "speech_pad_ms": 600}

print(f"loading {MODEL} on cuda...", flush=True)
t0 = time.time()
from faster_whisper import WhisperModel  # noqa: E402

model = WhisperModel(MODEL, device="cuda", compute_type="float16")
import numpy as np  # noqa: E402

model.transcribe(np.zeros(16000, dtype=np.float32))
print(f"model ready in {time.time() - t0:.1f}s", flush=True)


def decode_any(raw):
    from faster_whisper.audio import decode_audio
    return np.asarray(decode_audio(io.BytesIO(raw), sampling_rate=16000),
                      dtype=np.float32)


class H(BaseHTTPRequestHandler):
    server_version = "FlowNode/1.0"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            return self._json(200, {"ok": True, "model": MODEL, "node": "multivac-4080"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.headers.get("Authorization", "") != f"Bearer {TOKEN}":
            return self._json(401, {"error": "unauthorized"})
        if self.path.rstrip("/") != "/transcribe":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n)
        try:
            audio = decode_any(raw)
        except Exception as e:
            return self._json(400, {"error": f"bad audio: {e}"})
        t0 = time.time()
        prompt = (", ".join(v.strip() for v in VOCAB.split(",") if v.strip()) + "."
                  if VOCAB else None)
        segs, _ = model.transcribe(
            audio, language="en", vad_filter=True, vad_parameters=VAD,
            beam_size=3, condition_on_previous_text=False,
            no_speech_threshold=None, initial_prompt=prompt)
        text = " ".join(s.text.strip() for s in segs).strip()
        return self._json(200, {"text": text,
                                "asr_s": round(time.time() - t0, 2),
                                "duration_s": round(len(audio) / 16000, 2),
                                "node": "multivac-4080"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("ASR_TOKEN required")
    print(f"listening on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
