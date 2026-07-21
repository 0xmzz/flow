"""Desktop/window primitives and teach-loop behavior without real OS effects."""

import sys
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
import winctl  # noqa: E402
from commands import Router, _action_shaped, _parse_place, openable_app  # noqa: E402
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


def test_snapshot_snaps_regions_and_falls_back_to_relative_rects():
    rects = {1: (0, 0, 500, 1000), 2: (-900, 100, -300, 700),
             3: (0, 0, 10, 10)}

    class FakeUser32:
        def IsIconic(self, hwnd):
            return hwnd == 3

        def GetWindowRect(self, hwnd, pointer):
            rect = pointer._obj
            rect.left, rect.top, rect.right, rect.bottom = rects[hwnd]
            return True

    result = winctl.snapshot_layout(
        user32=FakeUser32(),
        windows_supplier=lambda: [(1, "Inbox - Google Chrome"),
                                  (2, "Odd Editor"), (3, "Minimized")],
        monitors=[(1, 0, 0, 1000, 1000, True),
                  (2, -1000, 0, 0, 1000, False)],
        process_resolver=lambda hwnd: {1: "chrome.exe", 2: "odd.exe", 3: "x.exe"}[hwnd])
    assert result == [
        {"exe": "chrome.exe", "title": "Inbox - Google Chrome",
         "monitor": 1, "region": "left"},
        {"exe": "odd.exe", "title": "Odd Editor", "monitor": 2,
         "do": "place_rect", "x": 100, "y": 100, "w": 600, "h": 600},
    ]


def test_place_rect_replays_monitor_relative_coordinates_with_fake_user32():
    calls = []

    class FakeUser32:
        def IsIconic(self, _handle):
            return False

        def MoveWindow(self, *args):
            calls.append(args)
            return True

    result = winctl.place_window_rect(
        "editor", 2, 100, 50, 600, 400, user32=FakeUser32(),
        windows_supplier=lambda: [(9, "Editor")],
        matcher=lambda _query, candidates: next(iter(candidates)),
        monitors=[(1, 0, 0, 1000, 1000, True),
                  (2, -1000, -200, 0, 800, False)])
    assert calls == [(9, -900, -150, 600, 400, True)]
    assert result["monitor"] == 2


def test_openable_app_builtin_dynamic_suffix_and_skip():
    apps = {"google chrome": "id1", "notepad plus plus": "id2"}
    assert openable_app("Code.exe", "thing", {}) == "visual studio code"
    assert openable_app("notepad-plus-plus.exe", "thing", apps) == "notepad plus plus"
    assert openable_app("host.exe", "Doc - Google Chrome", apps) == "google chrome"
    assert openable_app("mystery.exe", "Mystery", apps) is None


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

        def _place_rect_step(self, step):
            calls.append(("place_rect", dict(step)))

        def _desktop(self, arg):
            calls.append(("desktop", arg))

    recipe = {"name": "desk", "triggers": ["desk"], "steps": [
        {"do": "place", "arg": "chrome", "monitor": 2, "region": "left"},
        {"do": "place_rect", "arg": "editor", "monitor": 1,
         "x": -10, "y": 20, "w": 700, "h": 500},
        {"do": "desktop", "arg": "next"},
    ]}
    runner = WorkflowRunner(router=FakeRouter())
    assert runner.start(recipe)
    runner.thread.join(2)
    assert calls == [("place", recipe["steps"][0]),
                     ("place_rect", recipe["steps"][1]), ("desktop", "next")]


def test_layout_snapshot_builds_recipe_dedupes_and_reports_skips():
    root = ROOT / ".test-layout-snapshot"
    shutil.rmtree(root, ignore_errors=True)
    store = RecipeStore(root)
    snapshot = [
        {"exe": "chrome.exe", "title": "Inbox - Google Chrome", "monitor": 1,
         "region": "left"},
        {"exe": "chrome.exe", "title": "Docs - Google Chrome", "monitor": 2,
         "region": "right"},
        {"exe": "Code.exe", "title": "Flow - Visual Studio Code", "monitor": 2,
         "do": "place_rect", "x": 20, "y": 30, "w": 800, "h": 600},
        {"exe": "unknown.exe", "title": "Secret Tool", "monitor": 1,
         "region": "full"},
    ]
    router = Router({"cleanup": {}, "local_only": True}, workflow_store=store,
                    layout_supplier=lambda: snapshot, apps_supplier=lambda: {})
    response = router.handle("save this layout as focus mode")
    assert response.startswith("Saved 2 windows:")
    assert "Skipped: Secret Tool" in response
    recipe = store.get("focus mode")
    assert recipe["comment"] == "Captured from the current virtual desktop only."
    assert recipe["steps"] == [
        {"do": "open", "arg": "chrome"},
        {"do": "wait_window", "arg": "Inbox - Google Chrome", "timeout_s": 15},
        {"do": "place", "arg": "Inbox - Google Chrome", "monitor": 1,
         "region": "left"},
        {"do": "open", "arg": "visual studio code"},
        {"do": "wait_window", "arg": "visual studio code", "timeout_s": 15},
        {"do": "place_rect", "arg": "visual studio code", "monitor": 2,
         "x": 20, "y": 30, "w": 800, "h": 600},
    ]
    shutil.rmtree(root, ignore_errors=True)


def test_layout_name_collision_questions_before_replace():
    root = ROOT / ".test-layout-collision"
    shutil.rmtree(root, ignore_errors=True)
    store = RecipeStore(root)
    old = {"name": "desk", "triggers": ["desk"],
           "steps": [{"do": "open", "arg": "old"}]}
    store.save(old)
    router = Router(
        {"cleanup": {}, "local_only": True}, workflow_store=store,
        layout_supplier=lambda: [{"exe": "chrome.exe", "title": "Chrome",
                                  "monitor": 1, "region": "full"}],
        apps_supplier=lambda: {})
    question = router.handle("remember this layout as desk")
    assert isinstance(question, Question)
    assert question.handler("cancel") == "Layout save cancelled."
    assert store.get("desk") == old
    question = router.handle("save this setup as desk")
    assert question.handler("replace").startswith("Saved 1 windows:")
    assert store.get("desk")["steps"][0] == {"do": "open", "arg": "chrome"}
    shutil.rmtree(root, ignore_errors=True)


def test_layout_intent_action_dispatches_to_capture(monkeypatch):
    router = Router({"cleanup": {}, "local_only": True})
    monkeypatch.setattr(llm, "chain", lambda *args, **kwargs:
                        ('{"action":"save_layout","arg":"coding"}', "fake"))
    monkeypatch.setattr(router, "_save_layout", lambda name: f"captured {name}")
    assert router.handle("capture how my windows are arranged") == "captured coding"


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
