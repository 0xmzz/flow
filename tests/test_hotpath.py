"""Hotpath: a lost key-up, a busy press, or a clipboard blip must not eat a dictation.

Fast, no GPU, no network. Same shape as test_invariants.py: bare Flow instances,
everything external monkeypatched.
"""

import queue
import threading
import time
import types

import pytest


def _bare(monkeypatch):
    import flow
    f = flow.Flow.__new__(flow.Flow)
    f.cfg = {
        "hotkey": "right ctrl",
        "command_hotkey": "right alt",
        "live_hotkey": "f9",
        "hermes": {"enabled": False, "hotkey": "right shift"},
        "max_record_s": 600,
        "mode": "full",
        "streaming": {"enabled": False},
        "mute_output_while_recording": False,
        "sample_rate": 16000,
        "min_duration_s": 0.6,
    }
    f._pressed_keys = set()
    f._active_ptt_key = None
    f._key_lock = threading.Lock()
    f._key_seen_at = {}
    f._os_up_streak = 0
    f._record_started_at = None
    f._ctrl_q = queue.Queue()
    f._ctrl_stop = False
    f.recording = False
    f.active_mode = "dictate"
    f.streamer = None
    f.live_worker = None
    f.hermes = None
    f.hermes_context = None
    f.busy = threading.Lock()
    f._session_lock = threading.Lock()
    f.overlay = None
    f.muter = types.SimpleNamespace(mute=lambda: None, restore=lambda: None)
    f.recorder = types.SimpleNamespace(start=lambda: None, stop=lambda: None,
                                       level=0.0)
    monkeypatch.setattr(flow, "stop_speech", lambda: None)
    monkeypatch.setattr(flow.Flow, "set_icon", lambda self, *a, **k: None)
    return f


def _event(name, kind):
    import flow
    return types.SimpleNamespace(name=name, event_type=kind)


def _drain(f):
    """Run queued press/release actions the way _ctrl_loop does."""
    while True:
        try:
            action, mode = f._ctrl_q.get_nowait()
        except queue.Empty:
            return
        if action == "down":
            f.on_press(mode)
        elif action == "up":
            f.on_release()


def _hold(f, name="right ctrl"):
    f._pressed_keys.add(name)
    f._active_ptt_key = name
    f.recording = True
    f.active_mode = "dictate"


def test_max_record_s_default_is_ten_minutes():
    import flow
    assert flow.DEFAULTS["max_record_s"] == 600


def test_os_key_down_returns_none_when_the_query_fails(monkeypatch):
    """A missing user32 (non-Windows, or a broken call) degrades to None."""
    import ctypes
    import flow

    class _User32:
        @staticmethod
        def GetAsyncKeyState(vk):
            raise OSError("no user32")

    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(user32=_User32),
                        raising=False)
    assert flow._os_key_down(0xA3) is None


@pytest.mark.parametrize(("name", "vk"), [
    ("right ctrl", 0xA3),
    ("right alt", 0xA5),
    ("right shift", 0xA1),
    ("f9", 0x78),
])
def test_watchdog_queries_the_mapped_vk(name, vk, monkeypatch):
    import flow
    f = _bare(monkeypatch)
    f.cfg["hotkey"] = name
    _hold(f, name)
    seen = []
    monkeypatch.setattr(flow, "_os_key_down", lambda code: seen.append(code) or True)
    f._poll_key_watchdog()
    assert seen == [vk]
    assert f._active_ptt_key == name, "a key the OS still reports down stays held"


def test_watchdog_ignores_a_flicker_and_releases_on_the_second_poll(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    _hold(f)
    released = []
    monkeypatch.setattr(f, "on_release", lambda: released.append(1))
    down = {"v": False}
    monkeypatch.setattr(flow, "_os_key_down", lambda vk: down["v"])

    f._poll_key_watchdog()  # one UP poll is a flicker
    assert released == []
    assert f._active_ptt_key == "right ctrl"

    down["v"] = True  # OS says held again
    f._poll_key_watchdog()
    assert released == []
    assert f._os_up_streak == 0

    down["v"] = False
    f._poll_key_watchdog()
    assert released == []
    f._poll_key_watchdog()
    assert released == [1]
    assert f._active_ptt_key is None
    assert "right ctrl" not in f._pressed_keys


def test_watchdog_does_not_reconcile_an_unknown_hotkey(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    f.cfg["hotkey"] = "f8"
    _hold(f, "f8")
    seen = []
    monkeypatch.setattr(flow, "_os_key_down", lambda vk: seen.append(vk) or False)
    monkeypatch.setattr(f, "on_release", lambda: seen.append("rel"))
    f._poll_key_watchdog()
    f._poll_key_watchdog()
    assert seen == []
    assert f._active_ptt_key == "f8"


def test_watchdog_ignores_os_query_unavailable(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    _hold(f)
    monkeypatch.setattr(flow, "_os_key_down", lambda vk: None)
    monkeypatch.setattr(f, "on_release", lambda: (_ for _ in ()).throw(
        AssertionError("unavailable OS state must not synthesize a release")))
    f._poll_key_watchdog()
    f._poll_key_watchdog()
    assert f._active_ptt_key == "right ctrl"


def test_ctrl_loop_synthesizes_release_when_os_key_stays_up(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    _hold(f)
    released = []
    monkeypatch.setattr(f, "on_release", lambda: released.append(1))
    monkeypatch.setattr(flow, "_os_key_down", lambda vk: False)
    t = threading.Thread(target=f._ctrl_loop, daemon=True)
    t.start()
    deadline = time.time() + 2.0
    while not released and time.time() < deadline:
        time.sleep(0.02)
    f._ctrl_stop = True
    f._ctrl_q.put(("down", "dictate"))
    t.join(1.0)
    assert released, "a lost KEY_UP must be synthesized from the ctrl loop"
    assert f._active_ptt_key is None
    assert not t.is_alive()


def test_auto_repeat_inside_one_second_is_ignored(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    _hold(f)
    clock = {"t": 1000.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])
    f._key_seen_at["right ctrl"] = 1000.0
    clock["t"] = 1001.0  # exactly the repeat window
    f._on_key(_event("right ctrl", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.empty()
    assert f.recording is True
    assert f._active_ptt_key == "right ctrl"


def test_repress_after_a_missed_up_starts_a_new_recording(monkeypatch):
    """A KEY_DOWN >1s after the last event is a new press, not auto-repeat."""
    import flow
    f = _bare(monkeypatch)
    _hold(f)
    starts = []
    processed = []
    f.recorder = types.SimpleNamespace(start=lambda: starts.append(1),
                                       stop=lambda: "audio", level=0.0)

    class _Thread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None,
                     name=None):
            self.target = target
        def start(self):
            processed.append(getattr(self.target, "__name__", self.target))

    monkeypatch.setattr(flow.threading, "Thread", _Thread)
    clock = {"t": 5000.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])
    f._key_seen_at["right ctrl"] = 5000.0
    clock["t"] = 5001.2
    f._on_key(_event("right ctrl", flow.keyboard.KEY_DOWN))
    _drain(f)
    assert processed == ["process"], "the stuck dictation must be transcribed"
    assert starts == [1], "the new press must open a new recording"
    assert f.recording is True
    assert f._active_ptt_key == "right ctrl"


def test_on_press_stamp_feeds_the_local_orphan_cap(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    clock = {"t": 50.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])
    f.on_press("dictate")
    assert f.recording is True
    assert f._record_started_at == 50.0
    released = []
    monkeypatch.setattr(f, "on_release", lambda: released.append(1))
    clock["t"] = 50.0 + 599
    f._poll_record_cap()
    assert released == []
    clock["t"] = 50.0 + 600
    f._poll_record_cap()
    assert released == [1]


def test_max_record_s_processes_and_stops_the_mic(monkeypatch):
    """Over the cap the audio is handed to process(), not thrown away."""
    import flow
    f = _bare(monkeypatch)
    _hold(f)
    f._record_started_at = time.monotonic() - 601
    stopped = []
    f.recorder = types.SimpleNamespace(
        start=lambda: None,
        stop=lambda: stopped.append(1) or b"audio",
        level=0.0)
    processed = []

    class _Thread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None,
                     name=None):
            self.target = target
            self.args = args
        def start(self):
            processed.append(getattr(self.target, "__name__", ""))

    monkeypatch.setattr(flow.threading, "Thread", _Thread)
    f._poll_record_cap()
    assert stopped == [1]
    assert processed == ["process"]
    assert f.recording is False
    assert f._active_ptt_key is None
    assert "right ctrl" not in f._pressed_keys


def test_record_cap_does_not_steal_a_remote_session(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    _hold(f)
    f.active_mode = "remote"
    f._record_started_at = time.monotonic() - 1000
    monkeypatch.setattr(f, "on_release", lambda: (_ for _ in ()).throw(
        AssertionError("remote sessions keep their own cancel timer")))
    f._poll_record_cap()
    assert f.recording is True


def test_ctrl_loop_force_releases_a_local_recording_past_the_cap(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    _hold(f)
    f._record_started_at = time.monotonic() - 1000
    released = []
    monkeypatch.setattr(f, "on_release", lambda: released.append(1))
    # Key still physically down, so only the cap — not the watchdog — may fire.
    monkeypatch.setattr(flow, "_os_key_down", lambda vk: True)
    t = threading.Thread(target=f._ctrl_loop, daemon=True)
    t.start()
    deadline = time.time() + 2.0
    while not released and time.time() < deadline:
        time.sleep(0.02)
    f._ctrl_stop = True
    f._ctrl_q.put(("down", "dictate"))
    t.join(1.0)
    assert released, "max_record_s must be enforced from the ctrl loop"
    assert not t.is_alive()
