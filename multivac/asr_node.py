"""Flow fallback ASR node — runs on multivac's RTX 4080 (Laptop, 12GB).

Always-on, so phone/thin-client dictation survives beast being asleep, and
remote load stays off the actively-used 5090. Degraded on purpose: one-shot
transcription only (no streaming, no LLM cleanup) — dictation that always
works beats features that sometimes do.

Routes (Bearer token, same token as beast):
  GET  /health      -> {ok, model, node}
  POST /transcribe  -> wav/m4a/ogg body -> {text, asr_s, duration_s}
  POST /hermes      -> transcribe, then continue the most-recent Hermes session
"""

import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = os.environ.get("ASR_MODEL", "large-v3")
PORT = int(os.environ.get("ASR_PORT", "8757"))
TOKEN = os.environ.get("ASR_TOKEN", "")

VOCAB = os.environ.get("ASR_VOCAB", "")  # comma-separated, mirrors beast vocab
VAD = {"threshold": 0.3, "min_silence_duration_ms": 400, "speech_pad_ms": 600}
HERMES_URL = "http://127.0.0.1:8642"


def hermes_key(path=None):
    path = Path(path or "~/.hermes/.env").expanduser()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "API_SERVER_KEY":
                return value.strip().strip("\"'")
    except OSError:
        pass
    return ""


def hermes_request(method, path, key, body=None):
    data = (json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None else None)
    headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(HERMES_URL + path, data=data,
                                     headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def transcribe_audio(audio):
    t0 = time.time()
    prompt = (", ".join(v.strip() for v in VOCAB.split(",") if v.strip()) + "."
              if VOCAB else None)
    segs, _ = model.transcribe(
        audio, language="en", vad_filter=True, vad_parameters=VAD,
        beam_size=3, condition_on_previous_text=False,
        no_speech_threshold=None, initial_prompt=prompt)
    return (" ".join(s.text.strip() for s in segs).strip(),
            round(time.time() - t0, 2))

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
        route = self.path.rstrip("/")
        if route not in ("/transcribe", "/hermes"):
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n)
        try:
            audio = decode_any(raw)
        except Exception as e:
            return self._json(400, {"error": f"bad audio: {e}"})
        text, asr_s = transcribe_audio(audio)
        if route == "/hermes":
            key = hermes_key()
            if not key:
                return self._json(503, {"error": "Hermes is unavailable"})
            try:
                sessions = hermes_request(
                    "GET", "/api/sessions?limit=1&offset=0", key).get("data") or []
                if not sessions:
                    return self._json(503, {"error": "No Hermes sessions"})
                session = sessions[0]
                sid = str(session["id"])
                result = hermes_request(
                    "POST", "/api/sessions/" + sid + "/chat", key,
                    {"message": text})
                reply = ((result.get("message") or {}).get("content")) or ""
                return self._json(200, {
                    "text": text, "reply": reply,
                    "thread": session.get("title") or "Hermes",
                    "session_id": sid})
            except (OSError, ValueError, KeyError, urllib.error.HTTPError,
                    urllib.error.URLError):
                return self._json(502, {"error": "Hermes is unavailable"})
        return self._json(200, {"text": text,
                                "asr_s": asr_s,
                                "duration_s": round(len(audio) / 16000, 2),
                                "node": "multivac-4080"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("ASR_TOKEN required")
    print(f"listening on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
