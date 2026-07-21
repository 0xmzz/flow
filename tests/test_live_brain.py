"""Resourceful live-brain behavior, with every external surface faked."""

import datetime
import json
from pathlib import Path
import shutil

import pytest
import requests

import commands
import llm
from commands import OFFLINE_LINE, Router, machine_context
from workflows import Question, RecipeStore


NOW = datetime.datetime(2026, 7, 21, 8, 30)


@pytest.fixture
def brain_tmp(request):
    path = Path(__file__).resolve().parents[1] / "data" / "test-tmp" / request.node.name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _router(tmp_path, **kwargs):
    return Router(
        {"cleanup": {}, "local_only": True, "weather_location": "Dubai"},
        workflow_store=kwargs.pop("workflow_store", RecipeStore(tmp_path / "workflows")),
        now_fn=kwargs.pop("now_fn", lambda: NOW),
        state_path=tmp_path / "live_state.json",
        intent_log_path=tmp_path / "intent_log.jsonl",
        windows_supplier=kwargs.pop("windows_supplier", lambda: []),
        monitors_supplier=kwargs.pop("monitors_supplier", lambda: []),
        **kwargs)


def test_machine_context_composes_dedupes_and_hard_truncates():
    titles = [(index, "  Chrome   " + "x" * 90) for index in range(5)]
    titles += [(9, "chrome " + "x" * 90), (10, "Notes"), (11, "Mail")]
    value = machine_context(
        workflows=["Morning Setup", "Research"], workflow_running=True,
        agent_running=True, now=NOW, windows_supplier=lambda: titles,
        monitors_supplier=lambda: [object(), object()])
    assert value.startswith("Machine: Tuesday 08:30")
    assert value.lower().count("chrome") == 1
    assert "monitors: 2" in value and "Morning Setup" in value
    assert "running: workflow, agent" in value
    assert len(value) <= 400


def test_context_is_fresh_once_and_reused_by_intent_and_chat(brain_tmp, monkeypatch):
    seen = []
    windows = [[(1, "First")]]

    def fake_chain(_cfg, system, text, **kwargs):
        seen.append((system, text, kwargs.get("cloud_system")))
        return ('{"action":"answer","arg":""}' if kwargs["max_tokens"] == 80
                else "There is one window."), "local"

    monkeypatch.setattr(llm, "chain", fake_chain)
    router = _router(brain_tmp, windows_supplier=lambda: windows.pop())
    assert router.handle("what tabs are open") == "There is one window."
    assert len(seen) == 2
    assert all("windows: First" in system for system, _text, _cloud in seen)
    assert all("windows: First" not in text for _system, text, _cloud in seen)
    assert all("First" not in cloud for _system, _text, cloud in seen)


def test_morning_rule_fires_once_and_yes_runs_workflow(brain_tmp):
    store = RecipeStore(brain_tmp / "workflows")
    store.save({"name": "Morning Setup", "triggers": ["start my day"],
                "steps": []})
    router = _router(brain_tmp, workflow_store=store)
    assert router.claim_first_conversation_today()
    greeting = router.proactive_greeting(True)
    assert isinstance(greeting, Question)
    assert greeting.text == "Morning. Want me to run Morning Setup?"
    assert greeting.handler("yes") == "Running workflow Morning Setup."
    router.workflow_runner.thread.join(1)
    assert not router.claim_first_conversation_today()
    assert router.proactive_greeting(False) is None


def test_morning_rule_absent_workflow_is_plain_path(brain_tmp):
    router = _router(brain_tmp)
    assert router.claim_first_conversation_today()
    assert router.proactive_greeting(True) is None


class _Response:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        return None

    def json(self):
        return self.value


def test_weather_parsing_place_default_and_feed_format(brain_tmp, monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        if "geocoding" in url:
            return _Response({"results": [{"name": params["name"].title(),
                                            "latitude": 1.2, "longitude": 3.4}]})
        return _Response({"current": {"temperature_2m": 31.2,
                                      "apparent_temperature": 34.6,
                                      "weather_code": 1},
                          "daily": {"temperature_2m_max": [36.1],
                                    "temperature_2m_min": [27.4]}})

    monkeypatch.setattr(requests, "get", fake_get)
    router = _router(brain_tmp)
    paris = router.handle("weather in Paris")
    dubai = router.handle("what's the weather like?")
    assert paris == ("Paris is 31 degrees, feels like 35, mostly clear. "
                     "Today's high is 36 and low is 27.")
    assert dubai.startswith("Dubai is 31 degrees")
    assert [call[1]["name"] for call in calls if "geocoding" in call[0]] == [
        "paris", "Dubai"]
    assert all(call[2] == 6 for call in calls)


def test_weather_failure_line(brain_tmp, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k:
                        (_ for _ in ()).throw(requests.Timeout("nope")))
    assert _router(brain_tmp).handle("weather now") == (
        "The weather service didn't answer.")


def test_ollama_retries_connection_refused_once(monkeypatch):
    calls, sleeps = [], []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise requests.ConnectionError("[WinError 10061] actively refused")
        return _Response({"message": {"content": "ready"}})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    value = llm.call_provider({"provider": "ollama", "model": "q"}, "s", "t")
    assert value == "ready" and len(calls) == 2 and sleeps == [2]


def test_ollama_timeout_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k:
                        calls.append(1) or (_ for _ in ()).throw(requests.Timeout()))
    with pytest.raises(requests.Timeout):
        llm.call_provider({"provider": "ollama", "model": "q"}, "s", "t")
    assert calls == [1]


def test_warmup_keeps_retrying_until_success(monkeypatch):
    import flow
    posts, sleeps = [], []

    def fake_post(*args, **kwargs):
        posts.append(1)
        if len(posts) == 1:
            raise requests.ConnectionError("down")
        return _Response({})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(flow.time, "sleep", sleeps.append)
    instance = flow.Flow.__new__(flow.Flow)
    instance.cfg = {"cleanup": {"base_url": None, "model": "q"}}
    instance._warm_ollama()
    assert posts == [1, 1] and sleeps == [60]


def test_offline_line_and_intent_logs_for_llm_and_fastpath(brain_tmp, monkeypatch):
    router = _router(brain_tmp)
    monkeypatch.setattr(commands, "switch_to", lambda _name: "Chrome")
    assert router.handle("switch to chrome") == "Switched to Chrome."
    calls = []
    monkeypatch.setattr(llm, "chain", lambda *a, **k:
                        calls.append(1) or (None, None))
    assert router.handle("tell me a joke") == OFFLINE_LINE
    assert calls == [1]
    lines = [json.loads(line) for line in
             (brain_tmp / "intent_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["source"] == "fastpath" and lines[0]["action"] == "switch"
    assert lines[1]["source"] == "llm" and lines[1]["action"] == "offline"
    assert 0 < lines[1]["context_chars"] <= 400
