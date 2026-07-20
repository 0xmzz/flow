"""Invariants that must never regress. Fast, no GPU, no network.

Run: .venv\\Scripts\\python -m pytest tests -q
"""

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
from store import TranscriptStore, decode_wav, wav_bytes  # noqa: E402
from live import LiveWorker, wake_command  # noqa: E402


# ---------- llm.chain: nothing leaves the box ----------

CLOUD_FB = {"provider": "anthropic", "model": "glm", "base_url": "https://api.z.ai",
            "api_key": "k"}


def test_is_local_loopback_only():
    assert llm.is_local({"provider": "ollama", "base_url": "http://127.0.0.1:11434"})
    assert llm.is_local({"provider": "ollama", "base_url": None})
    assert not llm.is_local({"provider": "ollama", "base_url": "http://100.64.1.2:11434"})
    assert not llm.is_local({"provider": "anthropic", "base_url": "http://127.0.0.1"})


def test_local_only_drops_cloud_fallback(monkeypatch):
    calls = []

    def fake(p, system, text, mt=2048):
        calls.append(p.get("provider"))
        if p.get("provider") == "ollama":
            raise Exception("ollama down")
        return "CLOUD"

    monkeypatch.setattr(llm, "call_provider", fake)
    cfg = {"provider": "ollama", "model": "q", "fallback": dict(CLOUD_FB)}
    out, lane = llm.chain(cfg, "sys", "hi", local_only=True)
    assert out is None and lane is None
    assert calls == ["ollama"], "cloud lane must never be called under local_only"


def test_cloud_lane_gets_sanitized_prompt_when_allowed(monkeypatch):
    seen = {}

    def fake(p, system, text, mt=2048):
        seen[p.get("provider")] = system
        if p.get("provider") == "ollama":
            raise Exception("down")
        return "ok"

    monkeypatch.setattr(llm, "call_provider", fake)
    cfg = {"provider": "ollama", "model": "q", "fallback": dict(CLOUD_FB)}
    out, lane = llm.chain(cfg, "FULL window ctx", "hi",
                          cloud_system="GENERIC", local_only=False)
    assert out == "ok"
    assert seen["ollama"] == "FULL window ctx"
    assert seen["anthropic"] == "GENERIC", "window/app context must not egress"


# ---------- store: privacy flags ----------

def test_private_entry_stores_no_audio(tmp_path):
    import numpy as np
    st = TranscriptStore(tmp_path, archive_audio=True)
    audio = np.zeros(16000, dtype=np.float32)
    e_priv = st.add("secret words", source="remote", window="bareera|WhatsApp",
                    audio=audio, private=True)
    e_pub = st.add("my words", source="pc-hotkey", window="Claude", audio=audio)
    assert e_priv.get("audio") is None, "private entries must not archive audio"
    assert e_priv["private"] is True
    assert e_pub.get("audio"), "trusted entries keep the corpus"
    rows = st.read()
    assert [r.get("private") for r in rows] == [True, None]


def test_wav_roundtrip():
    import numpy as np
    a = (np.sin(np.linspace(0, 100, 16000)) * 0.3).astype(np.float32)
    b, sr = decode_wav(wav_bytes(a, 16000))
    assert sr == 16000 and abs(len(b) - len(a)) < 4
    assert float(np.abs(b[:100] - a[:100]).max()) < 0.01


@pytest.mark.parametrize(("heard", "expected"), [
    ("flow open youtube", "open youtube"),
    ("Flo, open youtube", "open youtube"),
    ("floh open youtube", "open youtube"),
    ("hello flow", None),
    ("flow", None),
])
def test_live_wake_gate(heard, expected):
    assert wake_command(heard, "flow") == expected


def test_live_wake_gate_can_be_disabled():
    assert wake_command("open youtube", None) == "open youtube"


def test_ignored_live_utterance_never_stores():
    class Recorder:
        def subscribe(self, maxsize=96):
            import queue
            return queue.Queue()
        def unsubscribe(self, _q): pass
    class Engine:
        provider = "CPU"
        def decode(self, _audio): return "hello there", 0.01
    class Vad:
        def reset(self): pass
        def probability(self, _frame): return 0.0
    class Store:
        def add(self, *_a, **_kw): raise AssertionError("ignored audio stored")
    class Router:
        def handle(self, _text): raise AssertionError("ignored audio routed")
    worker = LiveWorker(Recorder(), Router(), Store(), {
        "engine": "fake", "wake_word": "flow", "silence_ms": 280,
        "max_utterance_s": 12}, engine=Engine(), vad=Vad())
    worker.start()
    worker._decode_and_act(__import__("numpy").zeros(512), worker._generation)
    worker.close()


def test_live_module_has_no_network_client_import():
    source = (ROOT / "live.py").read_text(encoding="utf-8")
    assert "import requests" not in source and "urllib" not in source


# ---------- flow module helpers (no Flow() init: no GPU) ----------

def test_untrusted_remote_is_private():
    import flow as F
    cfg = json.loads(json.dumps(F.DEFAULTS))
    trusted = cfg["trusted_devices"]

    def is_private(source, window):
        dev = (window or "").split("|")[0].lower()
        return source == "remote" and not any(t in dev for t in trusted)

    assert is_private("remote", "BAREERA-PC|WhatsApp")
    assert not is_private("remote", "android-bubble")
    assert not is_private("pc-hotkey", "anything")


def test_describe_target_display_only():
    import flow as F
    assert F.describe_target("Dannis — WhatsApp").startswith("→ WhatsApp")
    assert F.describe_target("Inbox | Gmail").startswith("→ Gmail")
    assert F.describe_target("") == ""


# ---------- server contracts (stub flow, real HTTP) ----------

class StubFlow:
    def __init__(self, tmp):
        import flow as F
        self.cfg = json.loads(json.dumps(F.DEFAULTS))
        self.cfg["server"]["token"] = "testtok"
        self.store = TranscriptStore(tmp)
        self.started_at = time.time()
        self.last_pull = None
        self.model_lock = threading.Lock()

    def transcribe_and_store(self, audio, sample_rate=None, source="remote",
                             window=None, committed=("", 0)):
        return "stub text", {"asr_s": 0.01, "duration_s": 0.5}


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    from server import make_server
    stub = StubFlow(tmp_path_factory.mktemp("data"))
    httpd = make_server(stub, "127.0.0.1", 0, "testtok")
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", stub
    httpd.shutdown()


def _get(url, tok=None):
    req = urllib.request.Request(url)
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_health_open_history_gated(srv):
    base, _ = srv
    code, body = _get(base + "/health")
    assert code == 200 and b"ok" in body
    code, _b = _get(base + "/api/history")
    assert code == 401, "history must require the token"
    code, _b = _get(base + "/api/history", tok="testtok")
    assert code == 200


def test_openai_route_strict_shape(srv):
    import numpy as np
    base, _ = srv
    wav = wav_bytes(np.zeros(16000, dtype=np.float32), 16000)
    req = urllib.request.Request(base + "/v1/audio/transcriptions", data=wav,
                                 headers={"Authorization": "Bearer testtok",
                                          "Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=10) as r:
        out = json.loads(r.read())
    assert set(out.keys()) == {"text"}, \
        "OpenAI route must return ONLY {text} - extra keys get pasted into chats"
