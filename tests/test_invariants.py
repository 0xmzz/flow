"""Invariants that must never regress. Fast, no GPU, no network.

Run: .venv\\Scripts\\python -m pytest tests -q
"""

import json
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
from agents import (AgentCancelled, AgentError, AgentRunner, dispatch,
                    select_lane)  # noqa: E402
from store import TranscriptStore, decode_wav, wav_bytes  # noqa: E402
from live import LiveWorker, correct_live_text, wake_command  # noqa: E402
from commands import Router  # noqa: E402
from hud import HudState  # noqa: E402
from workflows import (Question, RecipeStore, WorkflowRunner,
                       creation_dialog, validate_recipe)  # noqa: E402


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
    ("Hey flow.", ""),
    ("okay Flo, open youtube", "open youtube"),
    ("hello flow", None),
    ("flow", ""),
])
def test_live_wake_gate(heard, expected):
    assert wake_command(heard, "flow") == expected


def test_live_wake_gate_can_be_disabled():
    assert wake_command("open youtube", None) == "open youtube"


class _FakeRecorder:
    def subscribe(self, maxsize=96):
        import queue
        return queue.Queue()
    def unsubscribe(self, _q): pass


class _FakeVad:
    def reset(self): pass
    def probability(self, _frame): return 0.0


class _FakeClock:
    def __init__(self): self.now = 100.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def test_live_wake_alone_awaits_and_routes_follow_up():
    import numpy as np
    class Engine:
        provider = "CPU"
        text = "Flow."
        def decode(self, _audio): return self.text, 0.01
    class RouterStub:
        def __init__(self): self.calls = []
        def handle(self, text): self.calls.append(text); return "done"
    engine, router, clock = Engine(), RouterStub(), _FakeClock()
    responses, spoken, heard, awaiting = [], [], [], []
    worker = LiveWorker(_FakeRecorder(), router, None, {
        "engine": "fake", "wake_word": "flow", "silence_ms": 280,
        "max_utterance_s": 12}, engine=engine, vad=_FakeVad(), clock=clock,
        response_fn=responses.append, speak_fn=spoken.append,
        heard_fn=lambda text, command: heard.append((text, command)),
        awaiting_fn=lambda active, deadline: awaiting.append((active, deadline)))
    worker.start()
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert worker.awaiting_command and responses == ["Yes?"] and spoken == ["Yes?"]
    assert worker.conversation_deadline == 145.0
    assert heard == [("Flow.", "")] and awaiting[-1] == (True, 108.0)
    engine.text = "open youtube"
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert router.calls == ["open youtube"] and not worker.awaiting_command
    assert heard[-1] == ("open youtube", "open youtube")
    worker.close()


def test_live_awaiting_command_expires_and_pause_cancels():
    import numpy as np
    class Engine:
        provider = "CPU"
        text = "flow"
        def decode(self, _audio): return self.text, 0.01
    class RouterStub:
        def handle(self, _text): raise AssertionError("expired command routed")
    ignored, clock, engine = [], _FakeClock(), Engine()
    worker = LiveWorker(_FakeRecorder(), RouterStub(), None, {
        "engine": "fake", "wake_word": "flow", "silence_ms": 280,
        "max_utterance_s": 12, "conversation_s": 0},
        engine=engine, vad=_FakeVad(), clock=clock,
        ignored_fn=ignored.append)
    worker.start()
    worker._decode_and_act(np.zeros(512), worker._generation)
    clock.advance(8.1)
    engine.text = "open youtube"
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert ignored == ["open youtube"] and not worker.awaiting_command
    engine.text = "flow"
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert worker.awaiting_command
    worker.pause("recorder-owner")
    assert not worker.awaiting_command
    worker.close()


def test_live_question_routes_bare_answer_but_wake_command_has_priority():
    import numpy as np
    class Engine:
        provider = "CPU"
        text = "flow begin dialog"
        def decode(self, _audio): return self.text, 0.01
    answers = []
    class RouterStub:
        workflow_runner = None
        def handle(self, text):
            if text == "begin dialog":
                return Question("Which one?", lambda answer:
                                answers.append(answer) or "Thanks.")
            return f"command {text}"
    engine, responses = Engine(), []
    worker = LiveWorker(_FakeRecorder(), RouterStub(), None, {
        "engine": "fake", "wake_word": "flow", "silence_ms": 280,
        "max_utterance_s": 12}, engine=engine, vad=_FakeVad(),
        response_fn=responses.append)
    worker.start()
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert worker.awaiting_input and responses[-1] == "Which one?"
    engine.text = "flow list workflows"
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert worker.awaiting_input and responses[-1] == "command list workflows"
    engine.text = "the first one"
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert answers == ["the first one"] and responses[-1] == "Thanks."
    assert not worker.awaiting_input
    worker.close()


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
        "max_utterance_s": 12, "conversation_s": 0}, engine=Engine(), vad=Vad())
    worker.start()
    worker._decode_and_act(__import__("numpy").zeros(512), worker._generation)
    worker.close()


def test_live_conversation_enters_resets_expires_and_pause_exits():
    import numpy as np
    class Engine:
        provider = "CPU"
        text = "flow hello"
        def decode(self, _audio): return self.text, 0.01
    class RouterStub:
        def __init__(self): self.calls = []
        def handle(self, text): self.calls.append(text); return "done"
    engine, router, clock = Engine(), RouterStub(), _FakeClock()
    conversation = []
    worker = LiveWorker(_FakeRecorder(), router, None, {
        "engine": "fake", "wake_word": "flow", "silence_ms": 280,
        "max_utterance_s": 12, "conversation_s": 45},
        engine=engine, vad=_FakeVad(), clock=clock,
        conversation_fn=lambda active, deadline:
        conversation.append((active, deadline)))
    worker.start()
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert router.calls == ["hello"] and worker.conversation_deadline == 145.0
    clock.advance(30)
    engine.text = "open youtube"
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert router.calls[-1] == "open youtube"
    assert worker.conversation_deadline == 175.0
    clock.advance(45)
    engine.text = "expired speech"
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert router.calls[-1] == "open youtube"
    engine.text = "flow back"
    worker._decode_and_act(np.zeros(512), worker._generation)
    worker.pause("recorder-owner")
    assert not worker.conversation_deadline and conversation[-1] == (False, 0.0)
    worker.resume("recorder-owner")
    engine.text = "flow again"
    worker._decode_and_act(np.zeros(512), worker._generation)
    worker.stop()
    assert not worker.conversation_deadline
    worker.close()


@pytest.mark.parametrize("phrase", ["stop listening", "go to sleep", "that's all"])
def test_live_conversation_exit_phrases(phrase):
    import numpy as np
    class Engine:
        provider = "CPU"
        text = "flow hello"
        def decode(self, _audio): return self.text, 0.01
    class RouterStub:
        def handle(self, _text): return "done"
    engine, clock, spoken = Engine(), _FakeClock(), []
    worker = LiveWorker(_FakeRecorder(), RouterStub(), None, {
        "engine": "fake", "wake_word": "flow", "silence_ms": 280,
        "max_utterance_s": 12, "conversation_s": 45},
        engine=engine, vad=_FakeVad(), clock=clock, speak_fn=spoken.append)
    worker.start()
    worker._decode_and_act(np.zeros(512), worker._generation)
    engine.text = phrase
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert spoken[-1] == "Going quiet." and not worker.conversation_deadline
    worker.close()


def test_live_conversation_swallows_filler_but_question_gets_it():
    import numpy as np
    class Engine:
        provider = "CPU"
        text = "flow start"
        def decode(self, _audio): return self.text, 0.01
    answers = []
    class RouterStub:
        def __init__(self): self.calls = []
        def handle(self, text):
            self.calls.append(text)
            if text == "ask":
                return Question("Ready?", lambda answer:
                                answers.append(answer) or "Thanks.")
            return "done"
    engine, router = Engine(), RouterStub()
    worker = LiveWorker(_FakeRecorder(), router, None, {
        "engine": "fake", "wake_word": "flow", "silence_ms": 280,
        "max_utterance_s": 12, "conversation_s": 45},
        engine=engine, vad=_FakeVad())
    worker.start()
    worker._decode_and_act(np.zeros(512), worker._generation)
    engine.text = "okay"
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert router.calls == ["start"]
    engine.text = "ask"
    worker._decode_and_act(np.zeros(512), worker._generation)
    engine.text = "yes"
    worker._decode_and_act(np.zeros(512), worker._generation)
    assert answers == ["yes"]
    worker.close()


def test_live_token_correction_uses_vocab_and_initial_wake_only():
    assert correct_live_text("workful", "flow", ["workflows"]) == "workflows"
    assert correct_live_text("Floats open YouTube", "flow") == "Flow open YouTube"
    assert correct_live_text("to youtube", "flow", ["go"]) == "to youtube"


def test_live_module_has_no_network_client_import():
    source = (ROOT / "live.py").read_text(encoding="utf-8")
    assert "import requests" not in source and "urllib" not in source


# ---------- HUD state (no Tk/display) ----------

def test_hud_expands_and_collapses_at_idle_boundary():
    hud = HudState()
    hud.set_enabled(True, now=10.0)
    assert hud.tick(17.999)
    assert not hud.tick(18.0)
    hud.utterance("flow open notes", command="open notes", now=20.0)
    assert hud.expanded
    hud.set_enabled(False, now=21.0)
    hud.utterance("late stale decode", command="oops", now=22.0)
    assert not hud.enabled and not hud.expanded


def test_hud_ignored_style_and_new_utterance_clear_interaction():
    hud = HudState()
    hud.set_enabled(True, now=1.0)
    hud.utterance("flow first", command="first", now=2.0)
    hud.append("extra", now=2.1)
    hud.utterance("background speech", ignored=True, now=3.0)
    assert hud.ignored and list(hud.thinking) == []
    assert hud.status == "Say ‘flow’ first"
    hud.utterance("flow second", command="second", now=4.0)
    assert not hud.ignored and "first" not in " ".join(hud.thinking)


def test_hud_line_buffer_trims_oldest_and_awaiting_expires():
    hud = HudState(max_lines=3)
    hud.set_enabled(True, now=1.0)
    for i in range(5):
        hud.append(f"line {i}", now=2.0 + i)
    assert list(hud.thinking) == ["line 2", "line 3", "line 4"]
    hud.set_status("Yes?", question=True, deadline=20.0, now=12.0)
    assert hud.question and hud.status_text(12.1) == "Yes?  8s"
    hud.tick(20.1)
    assert not hud.question


def test_hud_conversation_countdown_yields_to_question():
    hud = HudState()
    hud.set_enabled(True, now=1.0)
    hud.set_conversation(True, deadline=50.0, now=5.0)
    assert hud.status_text(5.1) == "Conversation  45s"
    hud.set_status("Which one?", question=True, deadline=20.0, now=6.0)
    assert hud.status_text(12.1) == "Which one?  8s"


# ---------- command router (no OS actions, model, or network) ----------

def _router():
    return Router({"cleanup": {}, "local_only": True})


def test_price_temporal_tail_is_removed(monkeypatch):
    import commands
    seen = []
    monkeypatch.setattr(commands, "_coingecko_price",
                        lambda q: seen.append(q) or ("Bitcoin", 1.0, 0.0))
    assert _router().handle("price of bitcoin right now").startswith("Bitcoin is")
    assert seen == ["bitcoin"]


def test_close_homophone_routes_to_close(monkeypatch):
    router = _router()
    seen = []
    monkeypatch.setattr(router, "_close", lambda arg: seen.append(arg) or "closed")
    assert router.handle("clothes youtube") == "closed"
    assert seen == ["youtube"]


def test_type_skill_pastes_and_restores(monkeypatch):
    import commands
    import keyboard
    import pyperclip
    copied, keys = [], []
    monkeypatch.setattr(pyperclip, "paste", lambda: "old clip")
    monkeypatch.setattr(pyperclip, "copy", copied.append)
    monkeypatch.setattr(keyboard, "send", keys.append)
    monkeypatch.setattr(commands.time, "sleep", lambda _s: None)
    assert _router().handle("write hello there") == ""
    assert copied == ["hello there", "old clip"] and keys == ["ctrl+v"]


def test_open_strips_venue_and_rescues_site_in_phrase(monkeypatch):
    import commands
    router = _router()
    seen, opened = [], []
    monkeypatch.setattr(commands, "switch_to", lambda _name: None)
    monkeypatch.setattr(commands, "open_app", lambda name:
                        seen.append(name) or ("YouTube" if name == "youtube" else None))
    assert router._open("YouTube in the browser") == "Opening YouTube."
    assert seen == ["YouTube"]
    monkeypatch.setattr(commands, "open_app", lambda _name: None)
    monkeypatch.setattr(commands, "installed_apps", lambda: {})
    monkeypatch.setattr(commands.os, "startfile", lambda _name:
                        (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(commands.webbrowser, "open", opened.append)
    assert router._open("to YouTube in the browser") == "Opening youtube."
    assert opened == [commands.KNOWN_SITES["youtube"]]


def test_compound_open_splits_only_open_command(monkeypatch):
    router, seen = _router(), []
    monkeypatch.setattr(router, "_open", lambda name:
                        seen.append(name) or f"opened {name}")
    assert router.handle("open chrome and open youtube") == (
        "opened chrome opened youtube")
    assert seen == ["chrome", "youtube"]
    seen.clear()
    assert router.handle("open chrome and youtube") == (
        "opened chrome opened youtube")
    assert seen == ["chrome", "youtube"]


def test_intent_json_dispatch_and_answer_fallback(monkeypatch):
    calls = []
    outputs = iter([('prefix {"action":"open","arg":"calculator"} suffix', "x"),
                    ('{"action":"answer","arg":""}', "x"),
                    ("A concise answer.", "x")])
    monkeypatch.setattr(llm, "chain", lambda *a, **kw:
                        calls.append(kw.get("max_tokens")) or next(outputs))
    router = _router()
    monkeypatch.setattr(router, "_open", lambda arg: f"opened {arg}")
    assert router.handle("put calculator on screen") == "opened calculator"
    assert router.handle("why is the sky blue") == "A concise answer."
    assert calls == [80, 80, 200]


def test_live_memory_is_bounded_expires_resets_and_excludes_questions(monkeypatch):
    clock = _FakeClock()
    router = Router({"cleanup": {}, "local_only": True}, clock=clock)
    monkeypatch.setattr(router, "handle", lambda text: f"reply {text}")
    for index in range(7):
        router.handle_live(str(index))
    assert [item[1] for item in router._conversation] == [str(i) for i in range(1, 7)]
    router.handle("stateless")
    assert len(router._conversation) == 6
    monkeypatch.setattr(router, "handle", lambda _text: Question("Which?", object()))
    router.handle_live("question")
    assert len(router._conversation) == 6
    clock.advance(120.1)
    assert router._conversation_text("now") == "User: now"
    monkeypatch.setattr(router, "handle", lambda _text: "again")
    router.handle_live("new")
    router.reset_conversation()
    assert not router._conversation


def test_live_fallback_context_reaches_intent_and_chat_and_affirmation_dispatches(monkeypatch):
    seen = []

    def fake_chain(_cfg, _system, text, **kwargs):
        seen.append((kwargs["max_tokens"], text))
        if kwargs["max_tokens"] == 200:
            return "Say: open chrome", "fake"
        if "User: yes do it" in text:
            return '{"action":"open","arg":"chrome"}', "fake"
        return '{"action":"answer","arg":""}', "fake"

    monkeypatch.setattr(llm, "chain", fake_chain)
    router = _router()
    router.remember_exchange("earlier", "Earlier reply")
    opened = []
    monkeypatch.setattr(router, "_open", lambda arg: opened.append(arg) or f"Opening {arg}.")
    assert router.handle_live("could you put a browser on screen") == "Say: open chrome"
    assert router.handle_live("yes do it") == "Opening chrome."
    assert opened == ["chrome"]
    affirmation = [text for tokens, text in seen
                   if tokens == 80 and "User: yes do it" in text][0]
    assert "User: could you put a browser on screen\nYou said: Say: open chrome" in affirmation
    first_chat = [text for tokens, text in seen if tokens == 200][0]
    assert "User: earlier\nYou said: Earlier reply" in first_chat
    assert first_chat.endswith("User: could you put a browser on screen")
    assert all(len(text) <= 800 for _tokens, text in seen)


@pytest.mark.parametrize(("spoken", "chord"), [
    ("new tab", "ctrl+t"), ("open a new tab", "ctrl+t"),
    ("reopen tab", "ctrl+shift+t"), ("next tab", "ctrl+tab"),
    ("previous tab", "ctrl+shift+tab"), ("refresh", "f5"),
    ("reload", "f5"), ("go back", "alt+left"),
    ("go forward", "alt+right"),
])
def test_browser_key_skills_are_blind_and_silent(monkeypatch, spoken, chord):
    import keyboard
    sent = []
    monkeypatch.setattr(keyboard, "send", sent.append)
    assert _router().handle(spoken) == ""
    assert sent == [chord]


def test_navigate_to_google_reaches_site_opening(monkeypatch):
    import commands
    opened = []
    monkeypatch.setattr(commands, "switch_to", lambda _name: None)
    monkeypatch.setattr(commands, "open_app", lambda _name: None)
    monkeypatch.setattr(commands.webbrowser, "open", opened.append)
    assert _router().handle("navigate to google") == "Opening google."
    assert opened == [commands.KNOWN_SITES["google"]]


# ---------- Phase B workflows (all OS effects injected) ----------

def test_workflow_store_round_trip_and_fuzzy_trigger():
    root = ROOT / ".test-workflows-phase-b"
    shutil.rmtree(root, ignore_errors=True)
    try:
        store = RecipeStore(root)
        recipe = {"name": "Morning Setup",
                  "triggers": ["morning setup", "start my morning"],
                  "steps": [{"do": "open", "arg": "chrome"}]}
        store.save(recipe)
        assert store.get("morning setup") == recipe
        assert store.match("morning set up")["name"] == "Morning Setup"
        assert store.match("completely unrelated") is None
        assert RecipeStore(root).list() == [recipe]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_workflow_runner_order_cancel_and_ask_yes_branch():
    calls, progress, completed = [], [], []
    runner = WorkflowRunner(
        primitives=lambda action, arg: calls.append((action, arg)),
        ask_fn=lambda question, cancel: calls.append(("ask", question)) or True)
    recipe = {"name": "desk", "triggers": ["desk"], "steps": [
        {"do": "open", "arg": "notes"},
        {"do": "ask", "arg": "Search too?", "yes": [
            {"do": "search", "arg": "today"}]},
        {"do": "type", "arg": "Ready"}]}
    assert runner.start(recipe, progress=lambda *args: progress.append(args),
                        complete=completed.append)
    runner.thread.join(2)
    assert calls == [("open", "notes"), ("ask", "Search too?"),
                     ("search", "today"), ("type", "Ready")]
    assert [item[:2] for item in progress] == [(1, 3), (2, 3), (3, 3)]
    assert completed == ["Workflow desk complete."]

    entered, release = threading.Event(), threading.Event()
    clock = _FakeClock()
    def fake_wait(delay):
        entered.set()
        release.wait(1)
        clock.advance(delay)
    runner = WorkflowRunner(wait_fn=fake_wait, clock=clock)
    waiting = {"name": "waiter", "triggers": ["waiter"],
               "steps": [{"do": "wait_s", "arg": 30},
                         {"do": "say", "arg": "too late"}]}
    done = []
    runner.start(waiting, complete=done.append)
    assert entered.wait(1)
    runner.cancel()
    release.set()
    runner.thread.join(2)
    assert done == ["Workflow waiter cancelled."]


def test_workflow_voice_creation_dialog_with_fake_intents():
    root = ROOT / ".test-workflow-dialog"
    shutil.rmtree(root, ignore_errors=True)
    try:
        store = RecipeStore(root)
        intents = {"open chrome": ("open", "chrome"),
                   "type Hello": ("type", "Hello")}
        question = creation_dialog("Demo", store, intents.get)
        assert isinstance(question, Question)
        question = question.handler("nonsense")
        assert isinstance(question, Question)
        question = question.handler("open chrome")
        assert isinstance(question, Question)
        question = question.handler("type Hello")
        response = question.handler("done")
        assert response == "Saved workflow Demo with 2 steps."
        assert store.get("demo")["steps"] == [
            {"do": "open", "arg": "chrome"},
            {"do": "type", "arg": "Hello"}]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_workflow_composition_progress_cycles_and_depth():
    root = ROOT / "data" / "test-tmp" / "composition"
    shutil.rmtree(root, ignore_errors=True)
    store = RecipeStore(root)
    calls, progress = [], []
    recipes = [
        {"name": "third", "triggers": ["third"],
         "steps": [{"do": "open", "arg": "three"}]},
        {"name": "second", "triggers": ["second"],
         "steps": [{"do": "open", "arg": "two"}, {"do": "run", "arg": "third"}]},
        {"name": "first", "triggers": ["first"],
         "steps": [{"do": "open", "arg": "one"}, {"do": "run", "arg": "second"}]},
    ]
    for recipe in recipes:
        store.save(recipe)
    runner = WorkflowRunner(recipe_store=store,
                            primitives=lambda action, arg: calls.append((action, arg)))
    done = []
    runner.start(recipes[-1], progress=lambda *item: progress.append(item),
                 complete=done.append)
    runner.thread.join(2)
    assert calls == [("open", "one"), ("open", "two"), ("open", "three")]
    assert [item[2] for item in progress] == [
        "open one", "second > open two", "second > third > open three"]
    assert done == ["Workflow first complete."]

    store.save({"name": "self", "triggers": ["self"],
                "steps": [{"do": "run", "arg": "self"}]})
    done = []
    runner.start(store.get("self"), complete=done.append)
    runner.thread.join(2)
    assert "recursive workflow: self > self" in done[0]

    store.save({"name": "cycle a", "triggers": ["cycle a"],
                "steps": [{"do": "run", "arg": "cycle b"}]})
    store.save({"name": "cycle b", "triggers": ["cycle b"],
                "steps": [{"do": "run", "arg": "cycle a"}]})
    done = []
    runner.start(store.get("cycle a"), complete=done.append)
    runner.thread.join(2)
    assert "recursive workflow: cycle a > cycle b > cycle a" in done[0]

    store.save({"name": "fourth", "triggers": ["fourth"],
                "steps": [{"do": "open", "arg": "four"}]})
    store.save({"name": "third deep", "triggers": ["third deep"],
                "steps": [{"do": "run", "arg": "fourth"}]})
    store.save({"name": "second deep", "triggers": ["second deep"],
                "steps": [{"do": "run", "arg": "third deep"}]})
    deep = {"name": "first deep", "triggers": ["first deep"],
            "steps": [{"do": "run", "arg": "second deep"}]}
    done = []
    runner.start(deep, complete=done.append)
    runner.thread.join(2)
    assert "maximum depth 3" in done[0]
    shutil.rmtree(root, ignore_errors=True)


def test_workflow_if_window_schema_and_branches():
    calls = []
    recipe = {"name": "window", "triggers": ["window"], "steps": [
        {"do": "if_window", "arg": "Visual Studio Code",
         "then": [{"do": "say", "arg": "yes"}],
         "else": [{"do": "say", "arg": "no"}]}]}
    assert validate_recipe(recipe)
    assert not validate_recipe({**recipe, "steps": [
        {"do": "if_window", "arg": "code", "else": []}]})
    runner = WorkflowRunner(title_supplier=lambda: ["notes - visual studio code"],
                            speak_fn=calls.append)
    runner.start(recipe)
    runner.thread.join(2)
    assert calls == ["yes"]
    calls.clear()
    runner = WorkflowRunner(title_supplier=lambda: ["calculator"],
                            speak_fn=calls.append)
    runner.start(recipe)
    runner.thread.join(2)
    assert calls == ["no"]


def test_workflow_agent_success_failure_cancel_and_source():
    root = ROOT / "data" / "test-tmp" / "workflow-agent"
    shutil.rmtree(root, ignore_errors=True)
    lane = _FakeLane("codex", result="done")
    agent = AgentRunner([lane], log_path=root / "runs.jsonl")
    recipe = {"name": "agents", "triggers": ["agents"],
              "steps": [{"do": "agent", "arg": "do work", "lane": "codex"}]}
    progress, done = [], []
    runner = WorkflowRunner(agent_runner=agent)
    runner.start(recipe, progress=lambda *x: progress.append(x), complete=done.append)
    runner.thread.join(2)
    assert done == ["Workflow agents complete."] and lane.calls == ["do work"]
    record = json.loads((root / "runs.jsonl").read_text().splitlines()[0])
    assert record["source"] == "workflow:agents"

    def fail(_task, _progress, _cancel):
        raise AgentError("broken")
    agent = AgentRunner([_FakeLane("codex", run_fn=fail)], log_path=root / "fail.jsonl")
    runner = WorkflowRunner(agent_runner=agent)
    done = []
    runner.start(recipe, complete=done.append)
    runner.thread.join(2)
    assert "failed: agent error: broken" in done[0]

    entered = threading.Event()
    def block(_task, _progress, cancel):
        entered.set()
        cancel.wait(1)
        raise AgentCancelled("cancelled")
    agent = AgentRunner([_FakeLane("codex", run_fn=block)], log_path=root / "cancel.jsonl")
    runner = WorkflowRunner(agent_runner=agent)
    done = []
    runner.start(recipe, complete=done.append)
    assert entered.wait(1)
    runner.cancel()
    runner.thread.join(2)
    assert done == ["Workflow agents cancelled."]
    shutil.rmtree(root, ignore_errors=True)


def test_workflow_creation_run_if_window_and_agent_confirmation():
    root = ROOT / "data" / "test-tmp" / "creation-new"
    shutil.rmtree(root, ignore_errors=True)
    store = RecipeStore(root)
    store.save({"name": "Morning Setup", "triggers": ["morning setup"],
                "steps": [{"do": "open", "arg": "chrome"}]})
    lane = _FakeLane("codex")
    agent = AgentRunner([lane], log_path=root / "runs.jsonl")
    parser = lambda text: ({"do": "agent", "arg": "fix it", "lane": "codex"}
                           if text == "agent" else ("open", "notes"))
    question = creation_dialog("composed", store, parser, agent_runner=agent)
    question = question.handler("run the morning setup workflow")
    question = question.handler("if notes is open, open notes")
    question = question.handler("agent")
    assert question.text == "Step sends a task to codex. Keep it?"
    question = question.handler("yes")
    response = question.handler("done")
    assert response == "Saved workflow composed with 3 steps."
    assert store.get("composed")["steps"] == [
        {"do": "run", "arg": "Morning Setup"},
        {"do": "if_window", "arg": "notes", "then": [{"do": "open", "arg": "notes"}]},
        {"do": "agent", "arg": "fix it", "lane": "codex"}]
    assert lane.calls == []
    shutil.rmtree(root, ignore_errors=True)


# ---------- Phase C agent lanes (all lanes are fakes) ----------

class _FakeLane:
    def __init__(self, name, available=True, result="done", run_fn=None):
        self.name = name
        self._available = available
        self.result = result
        self.run_fn = run_fn
        self.calls = []

    def available(self):
        return self._available

    def run(self, task, progress_fn, cancel_event):
        self.calls.append(task)
        if self.run_fn:
            return self.run_fn(task, progress_fn, cancel_event)
        return self.result


def test_agent_lane_registry_selection_and_unavailable_fallback():
    lanes = [_FakeLane("first", False), _FakeLane("second", True, "ok")]
    assert select_lane(lanes=lanes).name == "second"
    assert dispatch("task", lanes=lanes) == "ok"
    with pytest.raises(AgentError, match="first is not available"):
        dispatch("task", lane="first", lanes=lanes)
    with pytest.raises(AgentError, match="No agent lanes"):
        dispatch("task", lanes=[_FakeLane("off", False)])


def test_agent_confirmation_yes_runs_and_no_aborts():
    lane = _FakeLane("fake", result="Finished the requested work.")
    test_root = ROOT / "data" / "test-tmp"
    runner = AgentRunner([lane], log_path=test_root / "confirm.jsonl")
    router = _router()
    router.agent_runner = runner
    try:
        question = router._confirm_agent("Keep My Task Case")
        assert isinstance(question, Question)
        assert question.text == "Send to fake: Keep My Task Case?"
        assert question.handler("no") == "Okay, not sending."
        assert lane.calls == []
        question = router._confirm_agent("Keep My Task Case")
        assert question.handler("yes") == "Sending to fake."
        runner.thread.join(2)
        assert lane.calls == ["Keep My Task Case"]
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_agent_intent_returns_confirmation(monkeypatch):
    lane = _FakeLane("fake")
    router = _router()
    router.agent_runner = AgentRunner([lane],
                                      log_path=ROOT / "data" / "test-tmp" / "intent.jsonl")
    monkeypatch.setattr(llm, "chain", lambda *args, **kwargs:
                        ('{"action":"agent","arg":"Research This Topic"}', "fake"))
    question = router.handle("please investigate this thoroughly")
    assert isinstance(question, Question)
    assert question.text == "Send to fake: Research This Topic?"


def test_agent_progress_throttle_and_jsonl_log():
    root = ROOT / "data" / "test-tmp"
    shutil.rmtree(root, ignore_errors=True)
    clock = _FakeClock()

    def run(_task, progress, _cancel):
        progress("one")
        clock.advance(1)
        progress("two")
        clock.advance(1)
        progress("three")
        return "full result"

    lane = _FakeLane("fake", run_fn=run)
    progress, completed = [], []
    runner = AgentRunner([lane], log_path=root / "agent_runs.jsonl",
                         clock=clock)
    try:
        assert runner.start("test task", progress=lambda *x: progress.append(x),
                            complete=completed.append)
        runner.thread.join(2)
        assert progress == [("fake", "one"), ("fake", "three")]
        assert completed[0]["outcome"] == "ok"
        lines = (root / "agent_runs.jsonl").read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        assert record["lane"] == "fake" and record["task"] == "test task"
        assert record["outcome"] == "ok" and record["result_tail"] == "full result"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_agent_cancel_path_is_logged():
    root = ROOT / "data" / "test-tmp"
    entered = threading.Event()

    def run(_task, _progress, cancel):
        entered.set()
        assert cancel.wait(1)
        raise AgentCancelled("cancelled")

    runner = AgentRunner([_FakeLane("fake", run_fn=run)],
                         log_path=root / "agent_runs.jsonl")
    try:
        assert runner.start("long task")
        assert entered.wait(1)
        runner.cancel()
        runner.thread.join(2)
        record = json.loads((root / "agent_runs.jsonl").read_text().strip())
        assert record["outcome"] == "cancelled"
    finally:
        shutil.rmtree(root, ignore_errors=True)


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


# ---------- microphone silence guard ----------

def test_peak_level_distinguishes_dead_mic_from_whisper():
    import numpy as np
    from flow import SILENCE_PEAK, peak_level
    dead = np.full(16000, 3e-5, dtype=np.float32)      # measured: sleeping headset
    whisper = np.full(16000, 6e-3, dtype=np.float32)   # measured: quiet speech
    assert peak_level(dead) < SILENCE_PEAK <= peak_level(whisper)
    assert peak_level(np.zeros(0, dtype=np.float32)) == 0.0


def test_live_input_devices_excludes_loopback_and_duplicates(monkeypatch):
    import flow
    devices = [
        {"name": "Microphone (Razer)", "max_input_channels": 1},
        {"name": "Stereo Mix (Realtek)", "max_input_channels": 2},
        {"name": "PC Speaker (Realtek output)", "max_input_channels": 2},
        {"name": "Microphone (Razer)", "max_input_channels": 1},   # duplicate
        {"name": "Headphones", "max_input_channels": 0},           # output only
    ]
    monkeypatch.setattr(flow.sd, "query_devices", lambda *a, **k: devices)
    assert flow.live_input_devices() == [(0, "Microphone (Razer)")]


def test_resolve_device_matches_name_substring(monkeypatch):
    import flow
    devices = [
        {"name": "Microphone (HyperX Cloud Alpha)", "max_input_channels": 1},
        {"name": "Microphone (2- Razer Seiren V3 Mini)", "max_input_channels": 1},
    ]

    def query(*args, **kwargs):
        if kwargs.get("kind") == "input":
            return devices[0]
        return devices

    monkeypatch.setattr(flow.sd, "query_devices", query)
    assert flow.Recorder.resolve_device("razer seiren")[0] == 1
    # No match and None both fall back to the Windows default (index None).
    assert flow.Recorder.resolve_device("nonexistent")[0] is None
    assert flow.Recorder.resolve_device(None)[0] is None


def test_custom_sites_merge_over_builtins(monkeypatch):
    import commands
    opened = []
    router = commands.Router(
        {"cleanup": {}, "local_only": True,
         "sites": {"google": "https://example.test/search",
                   "personal": "https://personal.test"}})
    monkeypatch.setattr(commands, "switch_to", lambda _name: None)
    monkeypatch.setattr(commands, "open_app", lambda _name: None)
    monkeypatch.setattr(commands.webbrowser, "open", opened.append)
    assert router._open("google") == "Opening google."
    assert router._open("personal") == "Opening personal."
    assert router.sites["github"] == commands.KNOWN_SITES["github"]
    assert opened == ["https://example.test/search", "https://personal.test"]


def test_dashboard_hermes_integration_gate():
    from server import _history_page
    disabled = _history_page(hermes=False)
    enabled = _history_page(hermes=True)
    assert "Daily brief" not in disabled
    assert "/api/brief" not in disabled
    assert "Hermes sync" not in disabled
    assert "Hermes mouthpiece" not in disabled
    assert "/api/hermes/turns" not in disabled
    assert "Daily brief" in enabled
    assert "/api/brief" in enabled
    assert "Hermes sync" in enabled
    assert "Hermes mouthpiece" in enabled
    assert "/api/hermes/turns" in enabled


def test_cuda_absent_downgrades_whisper_config(monkeypatch):
    import types
    import flow
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(
        get_cuda_device_count=lambda: 0))
    instance = flow.Flow.__new__(flow.Flow)
    instance.cfg = {"device": "cuda", "compute_type": "float16"}
    instance._downgrade_cuda_if_unavailable()
    assert instance.cfg == {"device": "cpu", "compute_type": "int8"}


def test_setup_does_not_write_existing_config(tmp_path, monkeypatch):
    import flow
    path = tmp_path / "config.json"
    original = b'{"hotkey":"custom","personal":"keep"}\n'
    path.write_bytes(original)
    monkeypatch.setattr(flow, "CONFIG_PATH", path)
    monkeypatch.setattr(flow, "APP_DIR", tmp_path)
    monkeypatch.setattr(flow, "_setup_cuda_status",
                        lambda: (False, "not available"))
    monkeypatch.setattr(flow, "_setup_ollama", lambda: (False, []))
    monkeypatch.setattr(flow, "live_input_devices", lambda: [])
    flow.setup_machine()
    assert path.read_bytes() == original
