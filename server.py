"""HTTP server mode for Flow — turns the PC into a private ASR backend.

Phones (or any device) on the tailnet POST audio and get transcripts back, all
processed on the local GPU. Nothing leaves your network.

Routes:
  POST /transcribe                     wav body -> {"text": ...}
  POST /v1/audio/transcriptions        OpenAI-Whisper-compatible (multipart 'file')
  POST /record/start                   begin a host-mic recording (VM bridge)
  POST /record/stop                    end it -> {"text": ...} (typed by caller)
  POST /record/cancel                  abandon it (no transcription)
  GET  /api/history?limit=N            recent dictations as JSON
  GET  /api/stats                      totals
  GET  /                               browsable history page
  GET  /health                         liveness
Auth: Bearer token (server.token) on everything except /health and /.
"""

import json
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from store import decode_any


def _parse_multipart(body, content_type):
    """Parse multipart/form-data -> (first file part's bytes, text fields dict)."""
    m = re.search(r"boundary=([^;]+)", content_type)
    if not m:
        return None, {}
    boundary = ("--" + m.group(1).strip().strip('"')).encode()
    file_bytes, fields = None, {}
    for part in body.split(boundary):
        if b"Content-Disposition" not in part:
            continue
        idx = part.find(b"\r\n\r\n")
        if idx == -1:
            continue
        head, content = part[:idx], part[idx + 4:].rstrip(b"\r\n")
        if b"filename=" in head:
            if file_bytes is None:
                file_bytes = content
        else:
            nm = re.search(rb'name="([^"]+)"', head)
            if nm:
                fields[nm.group(1).decode()] = content.decode("utf-8", "replace")
    return file_bytes, fields


def _local_base_urls(port):
    """Every base URL this host plausibly answers on, hostname first.

    Includes interfaces a given client cannot reach (Hyper-V vSwitch, VPN) —
    that is fine and deliberate: the client probes candidates rather than
    trusting them, so an unreachable entry costs one parallel 2s probe and
    including it costs nothing. Filtering here would mean guessing which
    network the client is on, which is exactly what we cannot know."""
    import socket
    urls, seen = [], set()

    def add(hostpart):
        url = f"http://{hostpart}:{port}"
        if hostpart and url not in seen:
            seen.add(url)
            urls.append(url)

    try:
        name = socket.gethostname()
        add(name)
        for info in socket.getaddrinfo(name, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                add(ip)
    except OSError:
        pass
    return urls


def make_server(flow, host, port, token):
    from streaming import SessionManager
    streams = SessionManager(flow)

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
            if path == "/clientlog":
                # Unauthed on purpose: capture client crashes even when the
                # token/config is the very thing that's broken. Tailnet-only.
                import datetime
                from pathlib import Path
                try:
                    body = json.loads(self._body().decode("utf-8"))
                except Exception:
                    body = {"raw": "unparseable"}
                dev = self.headers.get("X-Device", "?")
                line = (f"{datetime.datetime.now().isoformat(timespec='seconds')} "
                        f"[{dev}] {body.get('tag', '')}: {body.get('message', body)}")
                if flow.store:
                    with (Path(flow.store.base) / "client.log").open(
                            "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                print("[clientlog] " + line[:500])
                return self._json(200, {"ok": True})
            if not self._authed():
                return self._json(401, {"error": "unauthorized"})
            device = self.headers.get("X-Device", "phone")
            if path == "/hermes":
                if not getattr(flow, "hermes", None):
                    return self._json(503, {"error": "Hermes is unavailable"})
                try:
                    audio, sr = decode_any(self._body())
                    if sr != flow.cfg["sample_rate"]:
                        from store import resample_linear
                        audio = resample_linear(audio, sr, flow.cfg["sample_rate"])
                    segs = flow.decode(audio, prompt=flow.bias_prompt())
                    text = " ".join(s.text.strip() for s in segs).strip()
                    picked = flow.hermes.resolve("", allow_current=False)
                    result = flow.hermes.converse(picked, text) if text else {}
                    return self._json(200, {
                        "text": text, "reply": result.get("reply", ""),
                        "thread": (picked or {}).get("title", ""),
                        "session_id": (picked or {}).get("id", "")})
                except Exception:
                    return self._json(502, {"error": "Hermes is unavailable"})
            if path.startswith("/stream/"):
                parts = path.split("/")  # /stream/start | /stream/<sid>/<op>
                if parts[2] == "start":
                    s = streams.start(device)
                    return self._json(200, {"sid": s.sid})
                sid, op = parts[2], (parts[3] if len(parts) > 3 else "")
                s = streams.get(sid)
                if not s:
                    return self._json(404, {"error": "no such stream"})
                if op == "audio":
                    committed = s.add_audio(self._body())
                    if committed:
                        s.partial_served = True
                    return self._json(200, {"committed": committed})
                if op == "finish":
                    streams.drop(sid)
                    try:
                        text = s.finish()
                        if not s.partial_served:
                            # replacements shift the typed prefix — skip if the
                            # client already live-typed committed text
                            import flow as flowmod
                            text = flowmod.apply_replacements(text, flow.cfg)
                    except Exception as e:
                        return self._json(500, {"error": f"finish failed: {e}"})
                    return self._json(200, {"text": text})
                if op == "cancel":
                    streams.drop(sid)
                    s.cancel()
                    return self._json(200, {"ok": True})
                return self._json(404, {"error": "unknown stream op"})
            if path == "/record/start":
                if flow.remote_start():
                    return self._json(200, {"ok": True})
                return self._json(409, {"error": "busy (hotkey session in progress?)"})
            if path == "/record/stop":
                try:
                    text = flow.remote_stop(window=device)
                except Exception as e:
                    return self._json(500, {"error": f"transcription failed: {e}"})
                if text is None:
                    return self._json(409, {"error": "no remote recording in progress"})
                return self._json(200, {"text": text})
            if path == "/record/cancel":
                return self._json(200, {"ok": bool(flow.remote_cancel())})
            if path == "/transcribe":
                return self._transcribe(self._body(), "remote", device)
            if path == "/brief":
                # multivac's sweep posts the daily brief back so the dashboard
                # can show it (beast otherwise never sees the distilled brief).
                try:
                    req = json.loads(self._body().decode("utf-8"))
                    from pathlib import Path
                    d = Path(flow.store.base) if flow.store else None
                    if d:
                        (d / "brief-latest.json").write_text(
                            json.dumps({"day": req.get("day", ""),
                                        "brief": req.get("brief", "")[:4000]}),
                            encoding="utf-8")
                    return self._json(200, {"ok": True})
                except Exception as e:
                    return self._json(400, {"error": str(e)})
            if path == "/llm":
                # Local-only LLM for trusted tailnet callers (e.g. the Hermes
                # sweep on multivac) — primary provider only, never the API
                # fallback: the caller chose this route for privacy.
                import llm as llmmod
                try:
                    req = json.loads(self._body().decode("utf-8"))
                    p = {k: v for k, v in flow.cfg["cleanup"].items()
                         if k != "fallback"}
                    out = llmmod.call_provider(
                        p, req.get("system", ""), req.get("text", ""),
                        max_tokens=int(req.get("max_tokens", 1024)))
                    return self._json(200, {"text": out})
                except Exception as e:
                    return self._json(502, {"error": str(e)})
            if path == "/command":
                try:
                    audio, sr = decode_any(self._body())
                except Exception as e:
                    return self._json(400, {"error": f"bad audio: {e}"})
                # NEVER styled, on any lane — the router matches literal
                # phrases, so a skill that rewords the utterance breaks every
                # command. Mirrors process_command on the host path.
                text, _ = flow.transcribe_and_store(
                    audio, sample_rate=sr, source="remote-command",
                    window=device, skip_cleanup=True)
                dev = (device or "").split("|")[0].lower()
                private = not any(
                    trusted in dev
                    for trusted in flow.cfg.get("trusted_devices", ()))
                response = flow.router.handle(text, private=private) if text else ""
                if not isinstance(response, str):  # dialogs are live-mode only
                    response = "That command needs a follow-up answer; not available remotely."
                return self._json(200, {"text": text, "response": response})
            if path == "/v1/audio/transcriptions":
                ctype = self.headers.get("Content-Type", "")
                body = self._body()
                if "multipart" in ctype:
                    audio, fields = _parse_multipart(body, ctype)
                else:
                    audio, fields = body, {}
                if not audio:
                    return self._json(400, {"error": "no audio file in request"})
                try:
                    a, sr = decode_any(audio)
                except Exception as e:
                    return self._json(400, {"error": f"bad audio: {e}"})
                text, _ = flow.transcribe_and_store(
                    a, sample_rate=sr, source="remote", window=device)
                # Strict OpenAI shape: plain text when asked, else {"text"} ONLY —
                # clients paste extra fields straight into the user's message.
                if fields.get("response_format") == "text":
                    raw = (text or "").encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                return self._json(200, {"text": text})
            return self._json(404, {"error": "not found"})

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "":
                hermes = bool(flow.cfg.get("integrations", {}).get("hermes"))
                return self._html(200, _history_page(hermes=hermes))
            if path == "/health":
                # Advertise every address this host answers on. A thin client
                # caches these and re-probes them later, so the host moving
                # (DHCP lease, tailnet expiry, new subnet) costs one failed
                # dictation instead of a manual config edit on someone else's
                # machine. Deliberately unauthenticated, like the rest of
                # /health — it leaks only addresses the client can already
                # reach, and gating it would defeat the discovery it exists for.
                return self._json(200, {"ok": True, "model": flow.cfg["model"],
                                        "addresses": _local_base_urls(port)})
            if path == "/client/setup.ps1":
                from pathlib import Path
                f = Path(__file__).parent / "client" / "setup.ps1"
                if not f.exists():
                    return self._json(404, {"error": "not found"})
                base = f"http://{self.headers.get('Host', '').split(':')[0]}:{flow.cfg['server']['port']}"
                body = (f.read_text(encoding="utf-8")
                        .replace("{{URL}}", base)
                        .replace("{{TOKEN}}", token or ""))
                data = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/client/version":
                import hashlib
                from pathlib import Path
                d = Path(__file__).parent
                h = hashlib.sha1()
                for f in ("flow.py", "store.py", "llm.py", "commands.py",
                          "supervisor.py", "workflows.py", "agents.py",
                          "winctl.py", "speech.py", "hud.py", "live.py",
                          "flow.ico"):
                    p = d / f
                    if p.exists():
                        h.update(p.read_bytes())
                return self._json(200, {"version": h.hexdigest()[:12]})
            if path.startswith("/client/file/"):
                from pathlib import Path
                name = path.split("/")[-1]
                allowed = {"flow.py", "store.py", "llm.py", "commands.py",
                           "supervisor.py", "workflows.py", "agents.py",
                           "winctl.py", "speech.py", "hud.py", "live.py",
                           "flow.ico"}
                if name not in allowed:
                    return self._json(404, {"error": "not found"})
                f = Path(__file__).parent / name
                data = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path in ("/vm/setup.sh", "/vm/daemon.py", "/vm/netwatch.sh"):
                from pathlib import Path
                f = Path(__file__).parent / "vm" / path.split("/")[-1]
                if not f.exists():
                    return self._json(404, {"error": "not found"})
                data = f.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/apk/version":
                from pathlib import Path
                meta = (Path(__file__).parent / "android" / "app" / "build"
                        / "outputs" / "apk" / "debug" / "output-metadata.json")
                if not meta.exists():
                    return self._json(404, {"error": "no build metadata"})
                m = json.loads(meta.read_text(encoding="utf-8"))
                el = (m.get("elements") or [{}])[0]
                return self._json(200, {"versionCode": el.get("versionCode", 0),
                                        "versionName": el.get("versionName", "?")})
            if path == "/apk":
                from pathlib import Path
                apk = (Path(__file__).parent / "android" / "app" / "build"
                       / "outputs" / "apk" / "debug" / "app-debug.apk")
                if not apk.exists():
                    return self._json(404, {"error": "apk not built"})
                data = apk.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type",
                                 "application/vnd.android.package-archive")
                self.send_header("Content-Disposition",
                                 'attachment; filename="flow.apk"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if not self._authed() and path.startswith("/api"):
                return self._json(401, {"error": "unauthorized"})
            if path == "/record/partial":
                # VM bridge polls this while the key is held and types the
                # committed delta live (committed text never retracts).
                if not self._authed():
                    return self._json(401, {"error": "unauthorized"})
                st = getattr(flow, "remote_streamer", None)
                return self._json(200, {"committed":
                                        st.committed_text if st else ""})
            if path == "/api/history":
                m = re.search(r"limit=(\d+)", self.path)
                limit = int(m.group(1)) if m else 200
                sm = re.search(r"since=([^&]+)", self.path)
                since = urllib.parse.unquote(sm.group(1)) if sm else None
                if since:
                    import datetime
                    flow.last_pull = datetime.datetime.now(
                        datetime.timezone.utc).isoformat(timespec="seconds")
                entries = flow.store.read(limit=limit, since=since)
                cursor = entries[-1]["ts"] if entries else since
                return self._json(200, {"entries": entries, "cursor": cursor})
            if path == "/api/hermes/turns":
                from hermes import read_turns
                m = re.search(r"limit=(\d+)", self.path)
                limit = int(m.group(1)) if m else 50
                base = getattr(getattr(flow, "store", None), "base", None)
                if base is None:
                    base = (getattr(flow, "hermes", None) or object())
                    turns_path = getattr(base, "turns_path", "")
                else:
                    from pathlib import Path
                    turns_path = Path(base) / "hermes_turns.jsonl"
                turns = read_turns(turns_path, limit)
                current = ""
                mouth = getattr(flow, "hermes", None)
                if mouth:
                    state = mouth.threads.load()
                    current = state.get("active") or state.get("current") or ""
                if not current and turns:
                    current = turns[-1].get("thread", "")
                return self._json(200, {"current_thread": current,
                                        "turns": turns})
            if path == "/api/brief":
                if not self._authed():
                    return self._json(401, {"error": "unauthorized"})
                from pathlib import Path
                f = (Path(flow.store.base) / "brief-latest.json") if flow.store else None
                if not f or not f.exists():
                    return self._json(200, {"day": "", "brief": ""})
                return self._json(200, json.loads(f.read_text(encoding="utf-8")))
            if path == "/api/meta":
                import time as _t
                from pathlib import Path
                base = Path(flow.store.base) if flow.store else None
                hist_b = audio_b = 0
                if base and (base / "history.jsonl").exists():
                    hist_b = (base / "history.jsonl").stat().st_size
                if base and (base / "dataset").exists():
                    audio_b = sum(f.stat().st_size
                                  for f in (base / "dataset").rglob("*.wav"))
                # fallback-node health, cached 60s so the dashboard poll is free
                import time as _tt
                cache = getattr(flow, "_node_cache", None)
                if not cache or _tt.time() - cache[0] > 60:
                    up = False
                    fb = flow.cfg.get("fallback_server_url", "")
                    if fb:
                        try:
                            import urllib.request as _ur
                            with _ur.urlopen(fb.rstrip("/") + "/health",
                                             timeout=2) as r:
                                up = b"ok" in r.read()
                        except Exception:
                            up = False
                    cache = (_tt.time(), up)
                    flow._node_cache = cache
                c = flow.cfg
                return self._json(200, {
                    "fallback_node": {"url": c.get("fallback_server_url", ""),
                                      "up": cache[1]},
                    "model": c["model"], "device": c["device"],
                    "streaming": c["streaming"]["enabled"],
                    "cleanup": {"provider": c["cleanup"]["provider"],
                                "model": c["cleanup"]["model"],
                                "fallback": bool(c["cleanup"].get("fallback", {}).get("model"))},
                    "hotkeys": {"dictate": c["hotkey"],
                                "command": c.get("command_hotkey")},
                    "uptime_s": int(_t.time() - flow.started_at),
                    "last_pull": flow.last_pull,
                    "storage": {"history_bytes": hist_b, "audio_bytes": audio_b,
                                "dir": str(base) if base else None},
                })
            if path == "/api/stats":
                return self._json(200, flow.store.stats())
            return self._json(404, {"error": "not found"})

        def log_message(self, *a):
            pass  # keep stdout clean

    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


def _history_page(hermes=False):
    # Self-contained dashboard; all data via the token-gated API.
    page = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Flow</title>
<style>
 :root{--bg:#0e0f13;--card:#16181d;--line:#22252c;--mut:#8a8f98;--fg:#e6e6e6;
       --acc:#3d7edb;--red:#dc3c3c;--amb:#f0b428;--grn:#3ec97e}
 body{font:15px/1.5 system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
 header{padding:16px 20px;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:22px;margin:0;font-weight:700;letter-spacing:.5px}
 .tag{color:var(--mut);font-size:13px}
 input{background:var(--card);border:1px solid var(--line);color:var(--fg);
       padding:6px 8px;border-radius:6px;margin-left:auto}
 .wrap{padding:0 20px 40px;max-width:1100px;margin:0 auto}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
 .card b{display:block;font-size:24px;font-weight:700}
 .card span{color:var(--mut);font-size:12px}
 .card small{color:var(--mut);font-size:11px;display:block;margin-top:2px}
 h2{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:1px;
    margin:26px 0 10px;font-weight:600}
 .bars{display:flex;gap:3px;align-items:flex-end;height:64px;background:var(--card);
       border:1px solid var(--line);border-radius:10px;padding:10px}
 .bars div{flex:1;background:var(--acc);border-radius:2px 2px 0 0;min-height:2px;opacity:.85}
 .kv{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:6px 14px}
 .kv div{display:flex;justify-content:space-between;padding:7px 0;
         border-bottom:1px solid var(--line);font-size:13px}
 .kv div:last-child{border-bottom:none}
 .kv b{font-weight:500}
 .kv span{color:var(--mut)}
 .ok{color:var(--grn)}
 .row{padding:11px 14px;border-bottom:1px solid var(--line)}
 .row:last-child{border-bottom:none}
 .rows{background:var(--card);border:1px solid var(--line);border-radius:10px}
 .meta{color:var(--mut);font-size:12px;margin-top:2px}
 .src{display:inline-block;padding:0 6px;border-radius:4px;background:#23262e;font-size:11px}
</style>
<header><h1>Flow</h1><span class=tag>your voice · your GPU · nothing leaves the box</span>
<input id=tok placeholder="token" size=14></header>
<div class=wrap>
 <div class=cards id=cards></div>
 <h2>Activity today (by hour)</h2><div class=bars id=bars></div>
 <h2>Daily brief <span style="text-transform:none;letter-spacing:0;color:var(--mut)">(distilled from your voice, on your GPU)</span></h2>
 <div class=kv id=brief style="white-space:pre-wrap;font-size:13px;padding:12px 14px">—</div>
 <h2>Pipeline &amp; data</h2><div class=kv id=kv></div>
 <section id=hermesSec><h2>Hermes mouthpiece</h2>
 <div class=rows id=hermesTurns></div></section>
 <h2>Recent dictations</h2><div class=rows id=list></div>
</div>
<script>
 const hm=location.hash.match(/tok=([^&]+)/);
 if(hm){localStorage.setItem('flow_tok',decodeURIComponent(hm[1]));history.replaceState(null,'',location.pathname)}
 const tokEl=document.getElementById('tok');tokEl.value=localStorage.getItem('flow_tok')||'';
 const esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
 const mb=b=>b>1048576?(b/1048576).toFixed(1)+' MB':Math.round(b/1024)+' KB';
 const ago=s=>{if(s==null)return 'never';const m=Math.round(s/60);
   return m<1?'just now':m<60?m+'m ago':Math.round(m/60)+'h ago'};
 function card(v,l,s){return '<div class=card><b>'+v+'</b><span>'+l+'</span>'+(s?'<small>'+s+'</small>':'')+'</div>'}
 async function load(){
  const t=tokEl.value;localStorage.setItem('flow_tok',t);
  const h={Authorization:'Bearer '+t};
  const [meta,d,br,ht]=await Promise.all([
    fetch('/api/meta',{headers:h}).then(r=>r.json()).catch(()=>({})),
    fetch('/api/history?limit=2000',{headers:h}).then(r=>r.json()).catch(()=>({entries:[]})),
    fetch('/api/brief',{headers:h}).then(r=>r.json()).catch(()=>({})),
    fetch('/api/hermes/turns?limit=30',{headers:h}).then(r=>r.json()).catch(()=>({turns:[]}))]);
  document.getElementById('brief').textContent=
    br.brief?(br.day?br.day+'\n':'')+br.brief:'— no brief yet (hourly sweep populates this)';
  const es=d.entries||[];
  const today=new Date().toLocaleDateString('sv');
  const te=es.filter(e=>new Date(e.ts).toLocaleDateString('sv')===today);
  const words=a=>a.reduce((n,e)=>n+(e.words||0),0);
  const secs=a=>a.reduce((n,e)=>n+(e.duration_s||0),0);
  const asr=te.filter(e=>e.asr_s!=null);
  const avg=asr.length?(asr.reduce((n,e)=>n+e.asr_s,0)/asr.length).toFixed(2)+'s':'—';
  const phone=te.filter(e=>(e.source||'').startsWith('remote')).length;
  document.getElementById('cards').innerHTML=
    card(te.length,'dictations today',words(te)+' words · '+Math.round(secs(te)/60)+' min spoken')+
    card(avg,'avg transcribe time','release → text')+
    card(phone,'from phone today',(te.length-phone)+' from PC')+
    card(es.length,'in history',words(es)+' words total');
  const hours=Array(24).fill(0);
  te.forEach(e=>{hours[new Date(e.ts).getHours()]++});
  const mx=Math.max(...hours,1);
  document.getElementById('bars').innerHTML=hours.map(c=>
    '<div style="height:'+Math.max(3,c/mx*100)+'%" title="'+c+'"></div>').join('');
  const st=meta.storage||{};
  document.getElementById('kv').innerHTML=
    '<div><b>ASR</b><span>'+esc(meta.model||'?')+' on '+esc(meta.device||'?')+
      (meta.streaming?' · streaming <span class=ok>on</span>':'')+'</span></div>'+
    '<div><b>Cleanup</b><span>'+esc((meta.cleanup||{}).model||'off')+' ('+esc((meta.cleanup||{}).provider||'')+
      ((meta.cleanup||{}).fallback?' + API fallback':'')+')</span></div>'+
    '<div><b>Hotkeys</b><span>'+esc((meta.hotkeys||{}).dictate||'')+' dictate · '+
      esc((meta.hotkeys||{}).command||'')+' command</span></div>'+
    '<div><b>Stored at</b><span>'+esc(st.dir||'?')+' · text '+mb(st.history_bytes||0)+
      ' · audio '+mb(st.audio_bytes||0)+'</span></div>'+
    '<div><b>Hermes sync</b><span>last pull '+(meta.last_pull?ago((Date.now()-new Date(meta.last_pull))/1000):'never')+
      ' · hourly from multivac</span></div>'+
    '<div><b>Fallback node</b><span>'+((meta.fallback_node||{}).up?'<span class=ok>up</span> — dictation survives this PC sleeping':'DOWN — phone/client dictation needs this PC on')+'</span></div>'+
    '<div><b>Uptime</b><span>'+((meta.uptime_s||0)<3600?Math.round((meta.uptime_s||0)/60)+' min':((meta.uptime_s||0)/3600).toFixed(1)+' h')+'</span></div>';
  document.getElementById('list').innerHTML=es.slice(-100).reverse().map(e=>
   '<div class=row><div>'+esc(e.text)+'</div><div class=meta><span class=src>'+
   esc(e.source||'')+'</span> '+esc((e.ts||'').replace('T',' ').slice(0,16))+
   ' · '+(e.duration_s||'?')+'s'+(e.asr_s!=null?' · asr '+e.asr_s+'s':'')+'</div></div>').join('');
  document.getElementById('hermesTurns').innerHTML=
    '<div class=row><b>Current thread: '+esc(ht.current_thread||'—')+'</b></div>'+
    (ht.turns||[]).slice().reverse().map(e=>'<div class=row><div>'+esc(e.text)+
      '</div><div class=meta><span class=src>'+esc(e.role||'')+'</span> '+
      esc(e.thread||'')+' · '+esc((e.ts||'').replace('T',' ').slice(0,16))+
      '</div></div>').join('');
 }
 tokEl.addEventListener('change',load);load();setInterval(load,30000);
</script>"""
    if hermes:
        return page
    page = re.sub(
        r"\s*<h2>Daily brief .*?</h2>\s*<div class=kv id=brief .*?</div>",
        "", page, flags=re.DOTALL)
    page = page.replace(
        "fetch('/api/brief',{headers:h}).then(r=>r.json()).catch(()=>({}))",
        "Promise.resolve({})")
    page = re.sub(r"\s*<section id=hermesSec>.*?</section>", "",
                  page, flags=re.DOTALL)
    page = page.replace(
        "fetch('/api/hermes/turns?limit=30',{headers:h}).then(r=>r.json()).catch(()=>({turns:[]}))",
        "Promise.resolve({turns:[]})")
    page = re.sub(
        r"\s*document\.getElementById\('hermesTurns'\)\.innerHTML=.*?;"
        r"(?=\s*})", "", page, flags=re.DOTALL)
    page = re.sub(
        r"\s*document\.getElementById\('brief'\)\.textContent=.*?;"
        r"(?=\s*const es=)",
        "", page, flags=re.DOTALL)
    page = re.sub(
        r"\s*'<div><b>Hermes sync</b>.*?</span></div>'\+",
        "", page, flags=re.DOTALL)
    return page
