import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes


class Response:
    def __init__(self, obj, status=200, lines=None):
        self.status = status
        self.body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return self.body

    def __iter__(self):
        return iter(self.lines or [])


def test_client_utf8_and_bearer(monkeypatch):
    seen = {}

    def open_(request, timeout):
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = request.data.decode("utf-8")
        return Response({"message": {"content": "Smart “quote” — yes"}})

    monkeypatch.setattr(hermes, "_OPENER", SimpleNamespace(open=open_))
    client = hermes.HermesClient(
        {"url": "http://hermes", "key": "secret", "timeout_s": 5})
    out = client.chat("abc", "hello — world")
    assert out["text"] == "Smart “quote” — yes"
    assert seen["auth"] == "Bearer secret"
    assert "hello — world" in seen["body"]


def test_client_fails_soft_without_leaking(monkeypatch):
    def open_(*args, **kwargs):
        raise OSError("secret internal detail")

    monkeypatch.setattr(hermes, "_OPENER", SimpleNamespace(open=open_))
    out = hermes.HermesClient(
        {"url": "http://x", "key": "do-not-print"}).health()
    assert out == {"ok": False, "error": hermes.ERROR}
    assert "secret" not in out["error"]


def test_sse_utf8(monkeypatch):
    lines = [
        'data: {"text":"one —"}\n'.encode(),
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]
    monkeypatch.setattr(
        hermes, "_OPENER",
        SimpleNamespace(open=lambda *a, **k: Response({}, lines=lines)))
    client = hermes.HermesClient({"url": "http://x", "key": "k"})
    assert list(client.chat_stream("s", "m")) == [{"text": "one —"}]


class FakeClient:
    def __init__(self, sessions=()):
        self.sessions = list(sessions)
        self.chats = []
        self.created = []
        self.runs = []

    def list_sessions(self, **kwargs):
        return {"ok": True, "data": {"data": list(self.sessions)}}

    def create_session(self, title):
        row = {"id": "new-id", "title": title}
        self.sessions.insert(0, row)
        self.created.append(title)
        return {"ok": True, "data": {"session": row}}

    def chat(self, sid, text):
        self.chats.append((sid, text))
        return {"ok": True, "text": "reply — smart"}

    def start_run(self, task):
        self.runs.append(task)
        return {"ok": True, "data": {"run_id": "r1"}}


def test_window_rule_then_recent_session(tmp_path):
    client = FakeClient([
        {"id": "recent", "title": "Phone"},
        {"id": "code", "title": "Builder"},
    ])
    mouth = hermes.Mouthpiece(
        {"thread_rules": [{"window": "Visual Studio CODE", "title": "builder"}]},
        tmp_path, client)
    assert mouth.resolve("foo visual studio code bar") == {
        "id": "code", "title": "Builder"}
    assert mouth.resolve("Telegram", allow_current=False) == {
        "id": "recent", "title": "Phone"}


def test_local_commands_do_not_chat_and_switch_is_durable(tmp_path):
    client = FakeClient([{"id": "a", "title": "Alpha"},
                         {"id": "b", "title": "Beta"}])
    mouth = hermes.Mouthpiece({}, tmp_path, client)
    result = mouth.converse({"id": "a", "title": "Alpha"}, "switch to beta")
    assert result["selection"] == {"id": "b", "title": "Beta"}
    assert mouth.resolve("no rule") == {"id": "b", "title": "Beta"}
    assert client.chats == []
    result = mouth.converse(None, "new thread Gamma")
    assert result["selection"]["title"] == "Gamma"
    assert client.chats == []


def test_turns_only_in_hermes_log_and_run_does_not_chat(tmp_path):
    client = FakeClient([{"id": "a", "title": "Alpha"}])
    mouth = hermes.Mouthpiece({}, tmp_path, client)
    result = mouth.converse({"id": "a", "title": "Alpha"}, "hello")
    assert result["reply"] == "reply — smart"
    rows = hermes.read_turns(tmp_path / "hermes_turns.jsonl")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert not (tmp_path / "history.jsonl").exists()
    result = mouth.converse({"id": "a", "title": "Alpha"},
                            "go build a tiny thing")
    assert result["run"] is True
    assert client.runs == ["a tiny thing"]
    assert client.chats == [("a", "hello")]


def test_right_ctrl_dispatch_never_queues_hermes(monkeypatch):
    import flow

    obj = object.__new__(flow.Flow)
    obj.cfg = {"hotkey": "right ctrl", "command_hotkey": "right alt",
               "live_hotkey": "f9",
               "hermes": {"enabled": True, "hotkey": "right shift"}}
    obj._pressed_keys = set()
    obj._active_ptt_key = None
    queued = []
    obj._ctrl_q = SimpleNamespace(put=queued.append)
    obj._on_key(SimpleNamespace(name="right ctrl",
                                event_type=flow.keyboard.KEY_DOWN))
    assert queued == [("down", "dictate")]


def test_dictation_transcript_never_sent_to_hermes(monkeypatch):
    import flow

    obj = object.__new__(flow.Flow)
    obj.cfg = json.loads(json.dumps(flow.DEFAULTS))
    obj.cfg["min_duration_s"] = 0
    obj.cfg["streaming"]["enabled"] = False
    obj.busy = __import__("threading").Lock()
    obj.live_worker = None
    obj.set_icon = lambda *a, **k: None
    obj.restore_base_state = lambda: None
    obj.mic_is_deaf = lambda audio: False
    obj.transcribe_and_store = lambda *a, **k: ("private dictation", {})
    obj.hermes = SimpleNamespace(
        converse=lambda *a, **k: pytest.fail("dictation reached Hermes"))
    pasted = []
    monkeypatch.setattr(flow, "paste_text",
                        lambda text, *a, **k: pasted.append(text))
    obj.process([0.1] * 16000)
    assert pasted == ["private dictation"]


def test_process_hermes_bypasses_history(monkeypatch, tmp_path):
    import flow

    obj = object.__new__(flow.Flow)
    obj.cfg = json.loads(json.dumps(flow.DEFAULTS))
    obj.cfg["hermes"]["enabled"] = True
    obj.cfg["min_duration_s"] = 0
    obj.busy = __import__("threading").Lock()
    obj.model_lock = __import__("threading").Lock()
    obj.overlay = None
    obj.live_worker = None
    obj.store = SimpleNamespace(add=lambda *a, **k: pytest.fail(
        "Hermes must not write dictation history"))
    obj.hermes = hermes.Mouthpiece(
        obj.cfg["hermes"], tmp_path,
        FakeClient([{"id": "a", "title": "Alpha"}]))
    obj.set_icon = lambda *a, **k: None
    obj.restore_base_state = lambda: None
    obj.mic_is_deaf = lambda audio: False
    obj.bias_prompt = lambda *a: None
    obj.decode = lambda *a, **k: [SimpleNamespace(text="hello")]
    monkeypatch.setattr(flow, "speak", lambda text: None)
    obj.process_hermes([0.1] * 16000, {"id": "a", "title": "Alpha"})
    assert not (tmp_path / "history.jsonl").exists()
    assert (tmp_path / "hermes_turns.jsonl").exists()


# --- Regression: the hotkey thread must stay local and unkillable ---------
# Both of these were real defects: resolve() ran inside on_press while holding
# _session_lock, so a hung Hermes froze right-ctrl dictation too, and one
# malformed session row raised KeyError and stranded self.recording=True,
# wedging every hotkey until restart.

def test_predict_title_makes_no_network_call():
    """on_press names the thread from local rules only — never over the wire."""
    class Exploding:
        def list_sessions(self, **kw):
            raise AssertionError("network call on the hotkey path")
        def create_session(self, *a, **kw):
            raise AssertionError("network call on the hotkey path")

    m = hermes.Mouthpiece(
        {"thread_rules": [{"window": "folionomics", "title": "Folionomics"}],
         "default_thread_title": "Flow"},
        tempfile.mkdtemp(), client=Exploding())
    assert m.predict_title("Folionomics — dashboard") == "Folionomics"
    assert m.predict_title("some other window") == "Flow"


def test_resolve_survives_session_row_without_id():
    """A malformed API row must not raise out onto the caller."""
    class Malformed:
        def list_sessions(self, **kw):
            return {"ok": True, "data": {"data": [
                {"title": "No id here", "last_active_at": "2026-07-27T00:00:00Z"},
                {"id": "good-1", "title": "Real", "last_active_at": "2026-07-26T00:00:00Z"},
            ]}}
        def create_session(self, *a, **kw):
            return {"ok": False}

    m = hermes.Mouthpiece({"thread_rules": []}, tempfile.mkdtemp(), client=Malformed())
    picked = m.resolve("anything")          # must not raise
    assert picked["id"] == "good-1"


def test_cheap_lookups_use_the_short_timeout():
    """A session list must never be able to hang for a full agent-turn timeout."""
    client = hermes.HermesClient({"url": "http://x", "key": "k", "timeout_s": 240})
    assert client.timeout == 240
    assert client.quick_timeout <= 8


def test_authenticated_calls_never_follow_redirects():
    """A 302 must not resend the Bearer key to the host the redirect names."""
    handler = hermes._NoRedirect()
    assert handler.redirect_request(
        SimpleNamespace(full_url="http://h/api/sessions"), None, 302, "Found",
        {}, "http://evil.example/api/sessions") is None
    assert any(isinstance(h, hermes._NoRedirect)
               for h in hermes._OPENER.handlers)
