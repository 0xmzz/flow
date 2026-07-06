"""HTTP server mode for Flow — turns the PC into a private ASR backend.

Phones (or any device) on the tailnet POST audio and get transcripts back, all
processed on the local GPU. Nothing leaves your network.

Routes:
  POST /transcribe                     wav body -> {"text": ...}
  POST /v1/audio/transcriptions        OpenAI-Whisper-compatible (multipart 'file')
  GET  /api/history?limit=N            recent dictations as JSON
  GET  /api/stats                      totals
  GET  /                               browsable history page
  GET  /health                         liveness
Auth: Bearer token (server.token) on everything except /health and /.
"""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from store import decode_any


def _parse_multipart_file(body, content_type):
    """Extract the first file part's bytes from a multipart/form-data body."""
    m = re.search(r"boundary=([^;]+)", content_type)
    if not m:
        return None
    boundary = ("--" + m.group(1).strip().strip('"')).encode()
    for part in body.split(boundary):
        if b"Content-Disposition" in part and b"filename=" in part:
            idx = part.find(b"\r\n\r\n")
            if idx != -1:
                return part[idx + 4:].rstrip(b"\r\n")
    return None


def make_server(flow, host, port, token):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Flow/1.0"

        def _authed(self):
            if not token:
                return True
            return self.headers.get("Authorization", "") == f"Bearer {token}"

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, code, text):
            body = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            return self.rfile.read(n) if n else b""

        def _transcribe(self, audio_bytes, source, window):
            try:
                audio, sr = decode_any(audio_bytes)
            except Exception as e:
                return self._json(400, {"error": f"bad audio: {e}"})
            text, meta = flow.transcribe_and_store(
                audio, sample_rate=sr, source=source, window=window)
            return self._json(200, {"text": text, **meta})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            if not self._authed():
                return self._json(401, {"error": "unauthorized"})
            device = self.headers.get("X-Device", "phone")
            if path == "/transcribe":
                return self._transcribe(self._body(), "remote", device)
            if path == "/v1/audio/transcriptions":
                ctype = self.headers.get("Content-Type", "")
                body = self._body()
                audio = (_parse_multipart_file(body, ctype)
                         if "multipart" in ctype else body)
                if not audio:
                    return self._json(400, {"error": "no audio file in request"})
                return self._transcribe(audio, "remote", device)
            return self._json(404, {"error": "not found"})

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path in ("", "/health"):
                return self._json(200, {"ok": True, "model": flow.cfg["model"]})
            if not self._authed() and path.startswith("/api"):
                return self._json(401, {"error": "unauthorized"})
            if path == "/api/history":
                m = re.search(r"limit=(\d+)", self.path)
                limit = int(m.group(1)) if m else 200
                return self._json(200, {"entries": flow.store.read(limit=limit)})
            if path == "/api/stats":
                return self._json(200, flow.store.stats())
            if path in ("", "/"):
                return self._html(200, _history_page())
            return self._json(404, {"error": "not found"})

        def log_message(self, *a):
            pass  # keep stdout clean

    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


def _history_page():
    # Self-contained page; fetches /api/history with a token entered once.
    return """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Flow history</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#0e0f13;color:#e6e6e6}
 header{padding:14px 16px;background:#16181d;position:sticky;top:0;display:flex;gap:10px;align-items:center}
 h1{font-size:16px;margin:0;font-weight:600}
 input{background:#0e0f13;border:1px solid #2a2d35;color:#e6e6e6;padding:6px 8px;border-radius:6px}
 .stat{color:#8a8f98;font-size:13px;margin-left:auto}
 .row{padding:12px 16px;border-bottom:1px solid #1c1e24}
 .meta{color:#6b7280;font-size:12px;margin-top:3px}
 .src{display:inline-block;padding:1px 6px;border-radius:4px;background:#23262e;font-size:11px}
</style>
<header><h1>Flow — dictation history</h1>
 <input id=tok placeholder="token" size=14>
 <span class=stat id=stat></span></header>
<div id=list></div>
<script>
 const tok=localStorage.getItem('flow_tok')||'';document.getElementById('tok').value=tok;
 function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
 async function load(){
  const t=document.getElementById('tok').value;localStorage.setItem('flow_tok',t);
  const h={Authorization:'Bearer '+t};
  const s=await fetch('/api/stats',{headers:h}).then(r=>r.json()).catch(()=>({}));
  document.getElementById('stat').textContent=(s.dictations||0)+' dictations · '+(s.words||0)+' words';
  const d=await fetch('/api/history?limit=500',{headers:h}).then(r=>r.json()).catch(()=>({entries:[]}));
  document.getElementById('list').innerHTML=(d.entries||[]).reverse().map(e=>
   '<div class=row><div>'+esc(e.text)+'</div><div class=meta><span class=src>'+
   esc(e.source||'')+'</span> '+esc(e.ts||'')+' · '+(e.duration_s||'?')+'s</div></div>').join('');
 }
 document.getElementById('tok').addEventListener('change',load);load();
</script>"""
