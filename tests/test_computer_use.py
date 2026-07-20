"""Desktop/window primitives and teach-loop behavior without real OS effects."""

import sys
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
import winctl  # noqa: E402
from commands import Router, _action_shaped, _parse_place  # noqa: E402
from workflows import Question, RecipeStore, WorkflowRunner  # noqa: E402


@pytest.mark.parametrize(("region", "expected"), [
    ("full", (-100, 10, 201, 101)),
    ("left", (-100, 10, 100, 101)),
    ("right", (0, 10, 101, 101)),
    ("top", (-100, 10, 201, 50)),
    ("bottom", (-100, 60, 201, 51)),
    ("top-left", (-100, 10, 100, 50)),
    ("top-right", (0, 10, 101, 50)),
    ("bottom-left", (-100, 60, 100, 51)),
    ("bottom-right", (0, 60, 101, 51)),
])
def test_region_math_preserves_odd_pixels_and_negative_coordinates(region, expected):
    assert winctl.region_rect((-100, 10, 101, 111), region) == expected


def test_place_window_uses_injected_windows_monitors_and_user32():
    calls = []

    class FakeUser32:
        def IsIconic(self, handle):
            return handle == 7

        def ShowWindow(self, *args):
            calls.append(("restore",) + args)
            return True

        def MoveWindow(self, *args):
            calls.append(("move",) + args)
            return True

    result = winctl.place_window(
        "chrome", monitor=2, region="left", user32=FakeUser32(),
        windows_supplier=lambda: [(7, "Inbox - Google Chrome")],
        matcher=lambda query, candidates: next(iter(candidates)),
        monitors=[(1, 0, 0, 100, 100, True),
                  (2, 100, 0, 301, 101, False)])
    assert result == {"title": "Inbox - Google Chrome", "monitor": 2,
                      "region": "left"}
    assert calls == [("restore", 7, 9), ("move", 7, 100, 0, 100, 101, True)]


@pytest.mark.parametrize(("spoken", "expected"), [
    ("put chrome on the left half of monitor two",
     {"do": "place", "arg": "chrome", "monitor": 2, "region": "left"}),
    ("move notion to screen 1 top right",
     {"do": "place", "arg": "notion", "monitor": 1, "region": "top-right"}),
    ("discord right half",
     {"do": "place", "arg": "discord", "monitor": None, "region": "right"}),
])
def test_place_phrase_parser(spoken, expected):
    assert _parse_place(spoken) == expected


def test_place_and_desktop_skills_dispatch_through_fakes(monkeypatch):
    router = Router({"cleanup": {}, "local_only": True})
    placed, keys = [], []
    monkeypatch.setattr(router, "_place_step",
                        lambda step: placed.append(step) or "placed")
    import keyboard
    monkeypatch.setattr(keyboard, "send", keys.append)
    assert router.handle("put chrome on the left of monitor two") == "placed"
    assert placed == [{"do": "place", "arg": "chrome", "monitor": 2,
                       "region": "left"}]
    assert router.handle("next desktop") == "Next desktop."
    assert router.handle("previous desktop") == "Previous desktop."
    assert keys == ["win+ctrl+right", "win+ctrl+left"]
    assert "can't reliably jump" in router.handle("desktop 2")
    assert keys == ["win+ctrl+right", "win+ctrl+left"]


def test_workflow_place_and_desktop_keep_structured_fields():
    calls = []

    class FakeRouter:
        def _place_step(self, step):
            calls.append(("place", dict(step)))

        def _desktop(self, arg):
            calls.append(("desktop", arg))

    recipe = {"name": "desk", "triggers": ["desk"], "steps": [
        {"do": "place", "arg": "chrome", "monitor": 2, "region": "left"},
        {"do": "desktop", "arg": "next"},
    ]}
    runner = WorkflowRunner(router=FakeRouter())
    assert runner.start(recipe)
    runner.thread.join(2)
    assert calls == [("place", recipe["steps"][0]), ("desktop", "next")]


def test_creation_step_parser_accepts_new_spoken_primitives():
    router = Router({"cleanup": {}, "local_only": True})
    assert router.parse_action("put chrome on the left of monitor two") == {
        "do": "place", "arg": "chrome", "monitor": 2, "region": "left"}
    assert router.parse_action("next desktop") == {"do": "desktop", "arg": "next"}
    assert router.parse_action("wait two seconds") == {"do": "wait_s", "arg": 2}


@pytest.mark.parametrize(("spoken", "expected"), [
    ("arrange my work apps", True), ("open everything for the day", True),
    ("can you arrange my apps", False), ("why is chrome open", False),
    ("what can you do", False),
])
def test_action_shaped_heuristic(spoken, expected):
    assert _action_shaped(spoken) is expected


def test_unknown_imperative_teaches_saves_and_replays_immediately(monkeypatch):
    root = ROOT / ".test-computer-use-workflows"
    shutil.rmtree(root, ignore_errors=True)
    store = RecipeStore(root)
    router = Router({"cleanup": {}, "local_only": True}, workflow_store=store)
    monkeypatch.setattr(llm, "chain", lambda *args, **kwargs:
                        ('{"action":"answer","arg":""}', "fake"))
    try:
        question = router.handle("arrange my work apps")
        assert isinstance(question, Question)
        assert question.text == "I don't know how to do that yet. Teach me the steps?"
        creation = question.handler("yes")
        assert isinstance(creation, Question)
        creation = creation.handler("open notion")
        response = creation.handler("done")
        assert response == "Saved. Say 'arrange my work apps' any time."
        recipe = store.match("arrange my work apps")
        assert recipe["triggers"] == ["arrange my work apps"]
        opened = []
        monkeypatch.setattr(router, "_open", lambda arg: opened.append(arg))
        assert router.handle("arrange my work apps").startswith("Running workflow")
        router.workflow_runner.thread.join(2)
        assert opened == ["notion"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unknown_imperative_no_falls_through_to_chat_once(monkeypatch):
    calls = []

    def fake_chain(*args, **kwargs):
        calls.append(kwargs["max_tokens"])
        return (("{\"action\":\"answer\",\"arg\":\"\"}" if
                 kwargs["max_tokens"] == 80 else "Chat answer."), "fake")

    monkeypatch.setattr(llm, "chain", fake_chain)
    question = Router({"cleanup": {}, "local_only": True}).handle("arrange my desk")
    assert isinstance(question, Question)
    assert question.handler("no") == "Chat answer."
    assert calls == [80, 200]
