import queue
import types

import keyboard

import flow


def _bare(recording=True, mode="dictate", summary=False):
    f = flow.Flow.__new__(flow.Flow)
    f.cfg = {"live_hotkey": None}
    f.recording = recording
    f.active_mode = mode
    f.summary_take = summary
    f._ctrl_q = queue.Queue()
    return f


def _ev(name, kind=keyboard.KEY_DOWN):
    return types.SimpleNamespace(name=name, event_type=kind)


def test_shift_during_dictation_turns_summary_on():
    f = _bare()
    f._on_key(_ev("right shift"))
    assert f._ctrl_q.get_nowait() == ("summary", None)


def test_shift_when_not_recording_does_nothing():
    f = _bare(recording=False)
    f._ptt_mode = lambda name: None
    f._on_key(_ev("shift"))
    assert f._ctrl_q.empty()


def test_shift_only_once_per_take():
    f = _bare(summary=True)
    f._ptt_mode = lambda name: None
    f._on_key(_ev("shift"))
    assert f._ctrl_q.empty()


def test_summary_failure_returns_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(flow, "_summary_call", boom)
    assert flow.llm_summarize("some long text here", {"server": {"token": "t"}}) is None
