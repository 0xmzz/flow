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
        "paste_mode": "clipboard",
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
    f._mic_lock = flow._new_mic_lock()
    f._mic_held = False
    f._job_q = queue.Queue()
    f._last_mic_probe = 0.0
    f._session_lock = threading.Lock()
    f.overlay = None
    f.tray = None
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
    f.recorder = types.SimpleNamespace(start=lambda: starts.append(1),
                                       stop=lambda: "audio", level=0.0)
    clock = {"t": 5000.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])
    f._key_seen_at["right ctrl"] = 5000.0
    clock["t"] = 5001.2
    f._on_key(_event("right ctrl", flow.keyboard.KEY_DOWN))
    _drain(f)
    job = f._job_q.get_nowait()
    assert job[0] == "dictate", "the stuck dictation must be transcribed"
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
    f._poll_record_cap()
    assert stopped == [1]
    assert f._job_q.get_nowait()[0] == "dictate"
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


def _busy_clipboard(monkeypatch):
    """A PyperclipWindowsException, including where ctypes has no WinError."""
    import ctypes
    import pyperclip
    if not hasattr(ctypes, "WinError"):
        monkeypatch.setattr(ctypes, "WinError",
                            lambda *a, **k: OSError("clipboard busy"),
                            raising=False)
    return pyperclip.PyperclipWindowsException("Error calling OpenClipboard")


def test_clipboard_copy_succeeds_on_the_third_try(monkeypatch):
    import pyperclip
    import flow
    copies = []
    presses = []
    writes = []

    def copy(text):
        copies.append(text)
        if len(copies) < 3:
            raise _busy_clipboard(monkeypatch)

    monkeypatch.setattr(pyperclip, "copy", copy)
    monkeypatch.setattr(flow.keyboard, "press", lambda key: presses.append(key))
    monkeypatch.setattr(flow.keyboard, "release", lambda key: None)
    monkeypatch.setattr(flow.keyboard, "write", lambda text: writes.append(text))
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    flow.paste_text("hello", "clipboard", restore=False)
    assert copies == ["hello", "hello", "hello"]
    assert writes == [], "a retry that lands must still paste, not retype"
    assert presses == ["ctrl", "v"]


def test_clipboard_failure_types_the_dictation(monkeypatch):
    import pyperclip
    import flow
    copies = []
    presses = []
    writes = []
    slept = []

    def copy(_text):
        copies.append(_text)
        raise pyperclip.PyperclipException("Error calling OpenClipboard")

    monkeypatch.setattr(pyperclip, "copy", copy)
    monkeypatch.setattr(flow.keyboard, "press", lambda key: presses.append(key))
    monkeypatch.setattr(flow.keyboard, "release", lambda key: None)
    monkeypatch.setattr(flow.keyboard, "write", lambda text: writes.append(text))
    monkeypatch.setattr(flow.time, "sleep", lambda s: slept.append(s))
    flow.paste_text("hello", "clipboard", restore=False)
    assert len(copies) == 10
    assert writes == ["hello"]
    assert presses == [], "a failed copy must not ctrl+v whatever is already there"
    assert sum(slept) == pytest.approx(0.30)


def test_typed_fallback_replaces_newlines_with_spaces(monkeypatch):
    """A typed fallback must not press Enter in the middle of a dictation."""
    import pyperclip
    import flow
    writes = []

    def copy(_text):
        raise pyperclip.PyperclipException("Error calling OpenClipboard")

    monkeypatch.setattr(pyperclip, "copy", copy)
    monkeypatch.setattr(flow.keyboard, "press", lambda key: None)
    monkeypatch.setattr(flow.keyboard, "release", lambda key: None)
    monkeypatch.setattr(flow.keyboard, "write", lambda text: writes.append(text))
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    flow.paste_text("line one\nline two\r\nline three", "clipboard", restore=False)
    assert writes == ["line one line two line three"]
    assert "\n" not in writes[0]

    writes.clear()
    flow.paste_text("line one\nline two", "type")
    assert writes == ["line one\nline two"]


def test_clipboard_restore_retries(monkeypatch):
    import pyperclip
    import flow
    calls = []

    def copy(text):
        calls.append(text)
        if text == "OLD" and calls.count("OLD") < 3:
            raise RuntimeError("Error calling OpenClipboard")

    monkeypatch.setattr(pyperclip, "copy", copy)
    monkeypatch.setattr(pyperclip, "paste", lambda: "OLD")
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(flow.keyboard, "press", lambda _k: None)
    monkeypatch.setattr(flow.keyboard, "release", lambda _k: None)

    class _Inline:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None,
                     name=None):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(flow.threading, "Thread", _Inline)
    flow.paste_text("NEW", "clipboard", restore=True)
    assert calls.count("NEW") == 1
    assert calls.count("OLD") == 3


def test_paste_exception_keeps_last_text_and_the_thread(monkeypatch):
    import numpy as np
    import flow
    f = _bare(monkeypatch)
    f.cfg["min_duration_s"] = 0
    f.cfg["paste_mode"] = "clipboard"
    says = []
    f.overlay = types.SimpleNamespace(set=lambda msg: says.append(msg))
    monkeypatch.setattr(flow.Flow, "mic_is_deaf", lambda self, audio: False)
    monkeypatch.setattr(flow.Flow, "restore_base_state", lambda self: None)
    monkeypatch.setattr(flow.Flow, "transcribe_and_store",
                        lambda self, audio, **kw: ("hello world", {}))

    def boom(text, mode, restore=True):
        assert f.last_text == "hello world", "remember the text before pasting"
        raise RuntimeError("Error calling OpenClipboard")

    monkeypatch.setattr(flow, "paste_text", boom)
    f.process(np.zeros(1600, dtype=np.float32))
    assert f.last_text == "hello world"
    assert says and says[0][0] == "say"
    assert "Copy last dictation" in says[0][1]


def test_tray_copies_the_last_dictation(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    f.cfg["server"] = {"port": 8756, "token": "t"}
    f.last_text = "kept sentence"
    copied = []
    monkeypatch.setattr(flow, "_copy_clipboard",
                        lambda text: copied.append(text) or True)
    items = dict(f._tray_menu_items())
    assert "Copy last dictation" in items
    items["Copy last dictation"]()
    assert copied == ["kept sentence"]


def test_mic_wait_is_one_second():
    import flow
    assert flow._MIC_WAIT_S == 1.0


def test_press_while_busy_records_and_pastes_in_order(monkeypatch):
    """A press during decode must record, and the two takes paste in order."""
    import flow
    f = _bare(monkeypatch)
    takes = iter(["first", "second"])
    starts = []
    f.recorder = types.SimpleNamespace(
        start=lambda: starts.append(1),
        stop=lambda: next(takes),
        level=0.0)
    pasted = []
    entered = threading.Event()
    release_first = threading.Event()

    def fake_process(self, audio, streamer=None, released_at=None, summary=False,
                     rewrite=False, full_samples=None, discarded=0,
                     committed_at_release=None, take_hwnd=0):
        self.busy.acquire()
        try:
            if audio == "first":
                entered.set()
                assert release_first.wait(3), "first take was never released"
            pasted.append(audio)
        finally:
            self.busy.release()

    monkeypatch.setattr(flow.Flow, "process", fake_process)
    worker = threading.Thread(target=f._job_loop, daemon=True)
    worker.start()
    f.recording = True
    f._mic_held = True
    f._mic_lock.acquire()
    f.on_release()
    assert entered.wait(2), "the first take should be decoding"
    assert f.busy.locked()
    f.on_press("dictate")
    assert f.recording is True, "a press while busy must start recording"
    assert starts == [1]
    f.on_release()
    release_first.set()
    deadline = time.time() + 2
    while pasted != ["first", "second"] and time.time() < deadline:
        time.sleep(0.01)
    f._job_q.put(None)
    worker.join(1)
    assert pasted == ["first", "second"]
    assert not worker.is_alive()


def test_press_during_a_short_probe_still_records(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    ready = threading.Event()

    def probe():
        f._mic_lock.acquire()
        ready.set()
        time.sleep(0.15)
        f._mic_lock.release()

    threading.Thread(target=probe, daemon=True).start()
    assert ready.wait(1)
    f.on_press("dictate")
    assert f.recording is True


def test_press_during_a_long_probe_is_not_silent(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    monkeypatch.setattr(flow, "_MIC_WAIT_S", 0.05)
    held = threading.Event()
    release = threading.Event()

    def probe():
        f._mic_lock.acquire()
        held.set()
        release.wait(2)
        f._mic_lock.release()

    threading.Thread(target=probe, daemon=True).start()
    assert held.wait(1)
    says = []
    f.overlay = types.SimpleNamespace(set=says.append)
    f.on_press("dictate")
    release.set()
    assert f.recording is False
    assert says and says[0][0] == "say"
    assert "Microphone" in says[0][1]


def test_startup_probe_does_not_hold_the_processing_lock(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    f.recorder.device_index = 1
    f.recorder.device_name = "Headset"
    seen = {}

    def probe(index, sample_rate, seconds=0.4):
        seen["busy"] = f.busy.locked()
        # Lock has no locked() we can rely on. A second thread failing to
        # acquire means the probe is holding the mic lock.
        grabbed = []

        def try_acquire():
            ok = f._mic_lock.acquire(blocking=False)
            grabbed.append(ok)
            if ok:
                f._mic_lock.release()

        t = threading.Thread(target=try_acquire)
        t.start()
        t.join()
        seen["mic_held"] = grabbed == [False]
        return 1.0

    monkeypatch.setattr(flow, "probe_input", probe)
    f.startup_mic_check()
    assert seen == {"busy": False, "mic_held": True}


def test_recover_mic_does_not_retarget_during_a_recording(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    f.recorder.device_name = "old"
    f.recorder.retarget = lambda spec: retargets.append(spec) or True
    retargets = []
    monkeypatch.setattr(flow, "live_input_devices", lambda: [(2, "New Mic")])
    monkeypatch.setattr(flow, "probe_input", lambda *a, **k: 1.0)
    f.recording = True
    f._mic_lock.acquire()
    done = threading.Event()

    def run():
        f.recover_mic(min_gap_s=0)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.15)
    assert retargets == []
    assert not done.is_set()
    f.recording = False
    f._mic_lock.release()
    assert done.wait(2)
    assert retargets == [2]


def test_remote_start_still_refuses_while_a_local_recording_is_active(monkeypatch):
    f = _bare(monkeypatch)
    f.recording = True
    f._mic_lock.acquire()
    assert f.remote_start() is False
    assert f.active_mode == "dictate"


def test_remote_stop_on_another_thread_frees_the_mic(monkeypatch):
    """remote_start's thread must not keep the mic after remote_stop.

    An RLock can only be released by its owner. remote_stop runs on the
    HTTP thread that did not acquire, the release is swallowed, and the
    next local press waits out the probe timeout forever.
    """
    import flow
    f = _bare(monkeypatch)
    f._remote_gen = 0
    monkeypatch.setattr(flow, "_MIC_WAIT_S", 0.2)
    f.recorder.stop = lambda: []
    says = []
    f.overlay = types.SimpleNamespace(set=says.append)
    assert f.remote_start() is True

    errors = []

    def stop():
        try:
            assert f.remote_stop(window="vm") == ""
        except Exception as exc:
            errors.append(exc)

    stopper = threading.Thread(target=stop)
    stopper.start()
    stopper.join(2)
    assert not stopper.is_alive()
    assert errors == []
    assert f.recording is False

    def press():
        f.on_press("dictate")

    presser = threading.Thread(target=press)
    presser.start()
    presser.join(2)
    assert not presser.is_alive()
    assert f.recording is True
    assert not any(isinstance(m, tuple) and m and m[0] == "say" for m in says)


def test_press_during_a_recording_does_not_wait_or_say_probe(monkeypatch):
    """A second key, or a local press during a remote take, returns at once."""
    import flow
    f = _bare(monkeypatch)
    monkeypatch.setattr(flow, "_MIC_WAIT_S", 0.4)
    says = []
    f.overlay = types.SimpleNamespace(set=says.append)
    f.on_press("dictate")
    assert f.recording is True
    started = time.monotonic()
    f.on_press("command")
    assert time.monotonic() - started < 0.2
    assert f.active_mode == "dictate"
    assert says == []

    f.on_release()
    assert f.recording is False
    held = threading.Event()
    release = threading.Event()

    def holder():
        f._mic_lock.acquire()
        f.recording = True
        held.set()
        release.wait(2)
        f.recording = False
        f._mic_lock.release()

    threading.Thread(target=holder, daemon=True).start()
    assert held.wait(1)
    started = time.monotonic()
    f.on_press("dictate")
    elapsed = time.monotonic() - started
    assert elapsed < 0.2
    assert f.recording is True
    assert says == []
    release.set()


def test_on_release_does_not_join_the_streamer(monkeypatch):
    """The ctrl loop must not wait out the streamer's up-to-2s join."""
    f = _bare(monkeypatch)
    f.recording = True
    joined = []

    class _Streamer:
        def halt(self):
            joined.append("halt")

        def stop(self):
            joined.append("stop")
            return "", 0

    f.streamer = _Streamer()
    f.on_release()
    assert joined == ["halt"]
    assert f._job_q.get_nowait()[0] == "dictate"


def test_paste_lands_before_history_and_the_final_decode_joins_first(monkeypatch):
    import types
    import numpy as np
    import flow
    f = _bare(monkeypatch)
    f.recording = True
    f.cfg["min_duration_s"] = 0
    f.cfg["cleanup"] = {"enabled": False}
    f.cfg["model"] = "m"
    f.cfg["device"] = "cpu"
    f.cfg["save_last_audio"] = True
    f.model_lock = threading.Lock()
    order = []

    class _Streamer:
        typed_len = 0

        def halt(self):
            order.append("halt")

        def stop(self):
            order.append("stop")
            return "hello", 0

        def restore_clipboard(self):
            return None

    f.streamer = _Streamer()
    f.recorder = types.SimpleNamespace(
        stop=lambda: np.zeros(1600, dtype=np.float32), level=0.0)
    f.store = types.SimpleNamespace(
        add=lambda *a, **k: order.append("store") or {})
    monkeypatch.setattr(flow.Flow, "mic_is_deaf", lambda self, audio: False)
    monkeypatch.setattr(flow.Flow, "decode",
                        lambda self, audio, **k: order.append("decode")
                        or [types.SimpleNamespace(text="there")])
    monkeypatch.setattr(flow, "paste_text", lambda *a, **k: order.append("paste"))
    monkeypatch.setattr(flow, "save_wav", lambda *a, **k: order.append("wav"))
    f.on_release()
    assert order == ["halt"]
    kind, args = f._job_q.get_nowait()
    assert kind == "dictate"
    f.process(*args)
    assert order == ["halt", "stop", "decode", "paste", "store", "wav"]


def test_transcribing_label_names_a_busy_pc_after_eight_seconds():
    import flow
    assert flow._transcribing_label(2.4) == "Transcribing… (2s)"
    assert flow._transcribing_label(7.9) == "Transcribing… (7s)"
    assert flow._transcribing_label(8.0) == "Still transcribing (8s) — the PC is busy"
    assert flow._transcribing_label(12.2) == "Still transcribing (12s) — the PC is busy"


def test_slow_process_emits_overlay_progress_and_stops(monkeypatch):
    import numpy as np
    import flow
    f = _bare(monkeypatch)
    f.cfg["min_duration_s"] = 0
    messages = []
    f.overlay = types.SimpleNamespace(set=messages.append)
    monkeypatch.setattr(flow.Flow, "mic_is_deaf", lambda self, audio: False)
    monkeypatch.setattr(flow, "_WAIT_HINT_S", 0.05)
    monkeypatch.setattr(flow, "_WAIT_TICK_S", 0.05)
    monkeypatch.setattr(flow, "_WAIT_BUSY_S", 0.12)

    def slow(self, audio, **kw):
        time.sleep(0.28)
        return "hello", {}

    monkeypatch.setattr(flow.Flow, "transcribe_and_store", slow)
    monkeypatch.setattr(flow, "paste_text", lambda *a, **k: None)
    f.process(np.zeros(1600, dtype=np.float32),
              released_at=time.monotonic() - 0.01)
    texts = [m[1] for m in messages
             if isinstance(m, tuple) and m and m[0] == "work"]
    assert any(t.startswith("Transcribing") for t in texts)
    assert any("the PC is busy" in t for t in texts)
    marked = len(messages)
    time.sleep(0.12)
    late = [m for m in messages[marked:]
            if isinstance(m, tuple) and m and m[0] == "work"]
    assert late == []


def test_wait_ticker_stays_quiet_while_a_newer_take_is_recording(monkeypatch):
    """Job A's 'Transcribing' ticks must not cover a newer take's overlay."""
    import flow
    f = _bare(monkeypatch)
    monkeypatch.setattr(flow, "_WAIT_HINT_S", 1.5)
    monkeypatch.setattr(flow, "_WAIT_TICK_S", 1.0)
    clock = {"t": 10.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])
    messages = []
    recording_at_push = []

    def capture(msg):
        messages.append(msg)
        recording_at_push.append(f.recording)

    f.overlay = types.SimpleNamespace(set=capture)
    f.recording = True
    phase = {"waits": 0}

    class _Stop:
        def is_set(self):
            return phase["waits"] >= 6

        def wait(self, timeout):
            phase["waits"] += 1
            if phase["waits"] == 4:
                f.recording = False
            clock["t"] += timeout
            return self.is_set()

    f._announce_wait(clock["t"], _Stop())
    work = [m for m in messages if isinstance(m, tuple) and m and m[0] == "work"]
    assert work, "ticks resume once the newer recording ends"
    assert recording_at_push == [False] * len(recording_at_push)


def test_fast_process_does_not_announce_a_wait(monkeypatch):
    import numpy as np
    import flow
    f = _bare(monkeypatch)
    f.cfg["min_duration_s"] = 0
    messages = []
    f.overlay = types.SimpleNamespace(set=messages.append)
    monkeypatch.setattr(flow.Flow, "mic_is_deaf", lambda self, audio: False)
    monkeypatch.setattr(flow.Flow, "transcribe_and_store",
                        lambda self, audio, **kw: ("hi", {}))
    monkeypatch.setattr(flow, "paste_text", lambda *a, **k: None)
    f.process(np.zeros(1600, dtype=np.float32))
    texts = [m[1] for m in messages
             if isinstance(m, tuple) and m and m[0] == "work"]
    assert texts == []


def _dictate(monkeypatch, f, released_at=None):
    """Run one local take through process() with the ASR stubbed out."""
    import types
    import numpy as np
    import flow
    f.cfg["min_duration_s"] = 0
    f.cfg["cleanup"] = {"enabled": False}
    f.cfg["model"] = "m"
    f.cfg["device"] = "cpu"
    f.model_lock = threading.Lock()
    monkeypatch.setattr(flow.Flow, "mic_is_deaf", lambda self, audio: False)
    def _decode(self, audio, **k):
        time.sleep(0.02)
        return [types.SimpleNamespace(text="hello")]

    monkeypatch.setattr(flow.Flow, "decode", _decode)
    monkeypatch.setattr(flow, "paste_text", lambda *a, **k: None)
    f.process(np.zeros(1600, dtype=np.float32), released_at=released_at)


def test_store_add_timing_fields_are_optional(tmp_path):
    from store import TranscriptStore
    st = TranscriptStore(tmp_path)
    plain = st.add("older line", source="pc-hotkey")
    assert "e2e_s" not in plain
    assert "stages" not in plain
    timed = st.add("newer", source="pc-hotkey", e2e_s=1.25,
                   stages={"decode": 1.0, "paste": 0.1})
    assert timed["e2e_s"] == 1.25
    assert timed["stages"]["decode"] == 1.0
    assert "e2e_s" not in st.read()[0]


def test_transcribe_route_omits_internal_timing(tmp_path):
    import json
    import urllib.request
    import numpy as np
    from server import make_server
    from store import TranscriptStore, wav_bytes

    class _Stub:
        def __init__(self):
            self.cfg = {"server": {"token": "tok"}}
            self.store = TranscriptStore(tmp_path)
            self.started_at = 0.0
            self.model_lock = threading.Lock()

        def transcribe_and_store(self, audio, sample_rate=None, source="remote",
                                 window=None, committed=("", 0)):
            return "hello", {
                "asr_s": 0.2, "duration_s": 1.0, "cleanup_s": 0.4,
                "pending": {"raw": "hello"},
            }

    httpd = make_server(_Stub(), "127.0.0.1", 0, "tok")
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        wav = wav_bytes(np.zeros(1600, dtype=np.float32), 16000)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/transcribe", data=wav,
            headers={"Authorization": "Bearer tok",
                     "Content-Type": "audio/wav"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            out = json.loads(resp.read())
    finally:
        httpd.shutdown()
    assert out == {"text": "hello", "asr_s": 0.2, "duration_s": 1.0}


def test_replacement_error_still_pastes_and_stores(monkeypatch, tmp_path, capsys):
    """A bad replacements entry must not drop the paste or the history row."""
    import types
    import numpy as np
    import flow
    from store import TranscriptStore
    f = _bare(monkeypatch)
    f.store = TranscriptStore(tmp_path)
    f.cfg["min_duration_s"] = 0
    f.cfg["cleanup"] = {"enabled": False}
    f.cfg["model"] = "m"
    f.cfg["device"] = "cpu"
    f.model_lock = threading.Lock()
    pasted = []
    monkeypatch.setattr(flow.Flow, "mic_is_deaf", lambda self, audio: False)
    monkeypatch.setattr(
        flow.Flow, "decode",
        lambda self, audio, **k: [types.SimpleNamespace(text="ship the report")])
    monkeypatch.setattr(flow, "paste_text",
                        lambda text, mode, restore=True: pasted.append(text))

    def boom(text, cfg):
        raise TypeError("bad replacement")

    monkeypatch.setattr(flow, "apply_replacements", boom)
    f.process(np.zeros(1600, dtype=np.float32))
    assert pasted == ["ship the report"]
    rows = f.store.read()
    assert len(rows) == 1
    assert rows[0]["text"] == "ship the report"
    assert "replacements failed" in capsys.readouterr().out


def test_history_row_carries_e2e_and_stages(monkeypatch, tmp_path):
    from store import TranscriptStore
    f = _bare(monkeypatch)
    f.store = TranscriptStore(tmp_path)
    _dictate(monkeypatch, f, released_at=time.monotonic() - 0.25)
    row = f.store.read()[-1]
    assert row["e2e_s"] >= 0.25
    for key in ("queue_wait", "stream_join", "decode", "cleanup", "paste"):
        assert key in row["stages"]
    assert row["stages"]["decode"] == row["asr_s"]
    assert row["text"] == "hello"


@pytest.mark.parametrize(("heard", "kept"), [
    ("Ship the report. Thanks for watching!", "Ship the report."),
    ("Ship the report. Thank you for watching.", "Ship the report."),
    ("Ship the report. Thank you so much for watching.", "Ship the report."),
    ("Thank you for watching, and I'll see you in the next video.", ""),
    ("See you in the next video.", ""),
    ("Please subscribe.", ""),
    ("Like and subscribe!", ""),
    ("Ship it. Bye. Bye. Bye. Bye. Bye.", "Ship it."),
    ("Ship it. Thank you. Thank you. Thank you.", "Ship it."),
    ("I walked the dog today. I walked the dog today. "
     "I walked the dog today.", "I walked the dog today."),
    ("I was watching the build fail earlier today.",
     "I was watching the build fail earlier today."),
    ("Do not subscribe to that theory, please think first.",
     "Do not subscribe to that theory, please think first."),
    ("That was kind. Thank you.", "That was kind. Thank you."),
    ("Go home. Go home.", "Go home. Go home."),
])
def test_strip_hallucinations(heard, kept):
    import flow
    assert flow.strip_hallucinations(heard) == kept


@pytest.mark.parametrize("text", [
    "Also Hermes uses GPT 5.5 right now with 272K context.",
    "It looks like 1.7 million worth of liquidity.",
    "It is ending in .67, so 192.168.68.67. Do you need the Mac?",
    "people are going to dev.folionomics, we should end it",
    "Snapshot, TimeSnap, MintSizing, etc., etc. How do we handle those?",
    "Can you re-look at the... ...consultation summary?",
    "Doesn't even look like it's... ...aligned.",
    "Opus 4.8 and GPT-5.6 on high thinking",
    "set to 25.5. The bedrooms can be 28",
    "https://app.hyperliquid.xyz is the site",
    "email me at a.b@c.com or atio.",
    "Thank you.",
    "Bye. Bye. Thank you.",
])
def test_strip_hallucinations_is_identity_when_nothing_is_removed(text):
    import flow
    assert flow.strip_hallucinations(text) == text


def test_strip_hallucinations_keeps_decimals_when_it_cuts_an_outro():
    import flow
    assert flow.strip_hallucinations(
        "Upgrade to GPT 5.5 please. Thanks for watching!"
    ) == "Upgrade to GPT 5.5 please."
    assert flow.strip_hallucinations(
        "people are going to dev.folionomics. Thanks for watching!"
    ) == "people are going to dev.folionomics."
    # Three or more dots in the kept sentence: the old splitter rejoined
    # those even after a real outro was removed.
    assert flow.strip_hallucinations(
        "It is ending in .67, so 192.168.68.67. Thanks for watching!"
    ) == "It is ending in .67, so 192.168.68.67."


def test_strip_hallucinations_removes_amara_credit_anywhere():
    import flow
    assert flow.strip_hallucinations(
        "a little bit more. Subtitles by the Amara.org community "
        "Messy as a strategy"
    ) == "a little bit more. Messy as a strategy"
    assert flow.strip_hallucinations(
        "Upgrade to GPT 5.5 please. Subtitles by the Amara.org community"
    ) == "Upgrade to GPT 5.5 please."
    assert flow.strip_hallucinations(
        "Subtitles by the Amara.org community"
    ) == ""


def test_filter_keeps_raw_history_and_does_not_touch_committed(monkeypatch, tmp_path):
    import types
    import numpy as np
    import flow
    from store import TranscriptStore
    f = _bare(monkeypatch)
    f.store = TranscriptStore(tmp_path)
    f.cfg["cleanup"] = {"enabled": False}
    f.cfg["model"] = "m"
    f.cfg["device"] = "cpu"
    f.cfg["trusted_devices"] = []
    f.model_lock = threading.Lock()
    monkeypatch.setattr(
        flow.Flow, "decode",
        lambda self, audio, **k: [types.SimpleNamespace(
            text="Send the file. Thanks for watching!")])
    text, _ = f.transcribe_and_store(
        np.zeros(1600, dtype=np.float32),
        committed=("Bye. Bye. Bye.", 0))
    assert text.startswith("Bye. Bye. Bye.")
    assert "Thanks for watching" not in text
    assert text.endswith("Send the file.")
    row = f.store.read()[-1]
    assert "Thanks for watching" in row["text"]
    assert row["cleaned"].endswith("Send the file.")
    assert "Thanks for watching" not in row["cleaned"]


def test_slow_e2e_prints_the_stage_breakdown(monkeypatch, capsys):
    f = _bare(monkeypatch)
    f.store = None
    _dictate(monkeypatch, f, released_at=time.monotonic() - 3.4)
    out = capsys.readouterr().out
    assert "[slow] e2e" in out
    assert "queue" in out and "join" in out
    assert "decode" in out and "paste" in out
    assert "s audio)" in out


def test_kokoro_not_warmed_in_dictation_only_config(monkeypatch):
    import inspect
    import speech
    import flow
    seen = []
    monkeypatch.setattr(speech, "configure",
                        lambda cfg, start=True: seen.append((cfg, start)))
    f = _bare(monkeypatch)
    f.cfg["tts"] = {"engine": "kokoro", "voice": "af_heart"}
    f.cfg["live"] = {"enabled": False}
    f.cfg["command_hotkey"] = None
    f.cfg["hermes"] = {"enabled": False}
    f._configure_speech()
    assert seen == [(f.cfg["tts"], False)]
    assert "_configure_speech" in inspect.getsource(flow.Flow.__init__)

    f.cfg["live"] = {"enabled": True}
    f._configure_speech()
    assert seen[-1][1] is True

    f.cfg["live"] = {"enabled": False}
    f.cfg["command_hotkey"] = "right alt"
    f._configure_speech()
    assert seen[-1][1] is True

    f.cfg["command_hotkey"] = ""
    f.cfg["hermes"] = {"enabled": True}
    f._configure_speech()
    assert seen[-1][1] is True


def test_silent_boot_reprobes_then_rechecks_once(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    f.recorder.device_index = 3
    f.recorder.device_name = "Headset"
    f.cfg["mic_autoswitch"] = True
    probes = []
    sleeps = []
    sweeps = []
    timers = []

    def probe(index, sample_rate, seconds=0.4):
        probes.append(index)
        return 0.0

    def recover(self, min_gap_s=20.0, when_silent="no live input found"):
        sweeps.append(when_silent)
        return None

    class _Timer:
        def __init__(self, interval, func):
            timers.append((interval, func))
            self._func = func

        def start(self):
            return None

    monkeypatch.setattr(flow, "probe_input", probe)
    monkeypatch.setattr(flow.Flow, "recover_mic", recover)
    monkeypatch.setattr(flow.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(flow.threading, "Timer", _Timer)
    f.startup_mic_check()
    assert probes == [3, 3, 3]
    assert sleeps == [flow._PINNED_REPROBE_S, flow._PINNED_REPROBE_S]
    assert sweeps == ["audio stack not ready"]
    assert len(timers) == 1 and timers[0][0] == flow._STACK_RECHECK_S
    timers[0][1]()
    assert probes == [3, 3, 3, 3]
    assert sweeps == ["audio stack not ready", "audio stack not ready"]
    assert len(timers) == 1, "the follow-up must not schedule another recheck"


def test_pinned_mic_that_wakes_on_the_second_probe_skips_the_sweep(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    f.recorder.device_index = 1
    f.recorder.device_name = "Headset"
    levels = iter([0.0, 1.0])
    monkeypatch.setattr(flow, "probe_input",
                        lambda *a, **k: next(levels))
    monkeypatch.setattr(flow.time, "sleep", lambda s: None)
    monkeypatch.setattr(flow.Flow, "recover_mic",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("a live pinned mic must not sweep")))
    f.startup_mic_check()


def test_startup_retry_gap_does_not_hold_the_mic_lock(monkeypatch):
    import flow
    f = _bare(monkeypatch)
    f.recorder.device_index = 1
    f.recorder.device_name = "Headset"
    monkeypatch.setattr(flow, "probe_input", lambda *a, **k: 0.0)

    def gap(seconds):
        assert f.busy.locked() is False
        assert f._mic_lock.acquire(blocking=False)
        f._mic_lock.release()

    monkeypatch.setattr(flow.time, "sleep", gap)
    monkeypatch.setattr(flow.Flow, "recover_mic", lambda *a, **k: "Other")
    monkeypatch.setattr(flow.threading, "Timer",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("a sweep that finds a mic does not recheck")))
    f.startup_mic_check()


def _hook_idle(monkeypatch, flow, seconds, parked):
    """Flag `parked` from whichever idle wait the streamer actually uses.

    A reverted `time.sleep` still trips the flag, then `stop` sits in `join`
    until its 2s timeout. `Event.wait` returns as soon as halt/stop/finish sets it.
    """
    real_sleep = flow.time.sleep

    def sleep(secs):
        if secs == seconds:
            parked.set()
        return real_sleep(secs)

    real_wait = threading.Event.wait

    def wait(self, timeout=None):
        if timeout == seconds:
            parked.set()
        return real_wait(self, timeout)

    monkeypatch.setattr(flow.time, "sleep", sleep)
    monkeypatch.setattr(threading.Event, "wait", wait)


def _stream_flow(monkeypatch, interval):
    import numpy as np
    import flow
    f = _bare(monkeypatch)
    f.model_lock = threading.Lock()
    f.cfg["sample_rate"] = 16000
    f.cfg["streaming"] = {
        "interval_s": interval,
        "unstable_tail_s": 0.6,
        "pause_gap_s": 0.25,
    }
    audio = np.ones(16000 * 2, dtype=np.float32)
    f.recorder = types.SimpleNamespace(snapshot=lambda: audio)
    return flow, f


def _agreed_words():
    """Stable prefix with pause gaps, so a pass that is allowed to finish commits."""
    return [types.SimpleNamespace(words=[
        types.SimpleNamespace(word=" hello", start=0.0, end=0.4),
        types.SimpleNamespace(word=" there", start=0.9, end=1.2),
    ])]


def test_stream_worker_halt_cuts_the_idle_wait(monkeypatch):
    flow, f = _stream_flow(monkeypatch, 5.0)
    parked = threading.Event()
    _hook_idle(monkeypatch, flow, 5.0, parked)
    worker = flow.StreamWorker(f)
    worker.start()
    assert parked.wait(1.0), "idle wait never started"
    worker.halt()
    t0 = time.perf_counter()
    text, samples = worker.stop()
    elapsed = time.perf_counter() - t0
    alive = worker._thread.is_alive()
    assert elapsed < 0.1 and not alive, (elapsed, alive)
    assert (text, samples) == ("", 0)


def test_agreeing_pass_still_commits_when_the_take_stays_open(monkeypatch):
    flow, f = _stream_flow(monkeypatch, 0.01)
    monkeypatch.setattr(flow.Flow, "decode",
                        lambda self, audio, **k: _agreed_words())
    worker = flow.StreamWorker(f)
    worker._prev = ["hello", "there"]
    worker.start()
    deadline = time.time() + 2
    while not worker.committed_text and time.time() < deadline:
        time.sleep(0.01)
    worker.halt()
    worker.stop()
    assert worker.committed_text == "hello there"
    assert worker.committed_samples == int(1.2 * 16000)
    assert not worker._thread.is_alive()


def _client_flow(monkeypatch):
    import numpy as np
    import flow
    f = _bare(monkeypatch)
    f.cfg["sample_rate"] = 16000
    f.cfg["server"] = {"token": "tok"}
    f.cfg["client_name"] = "box"
    f.cfg["server_url"] = "http://lane.test"
    f.resolve_lane = lambda *a, **k: "http://lane.test"
    f.recorder = types.SimpleNamespace(
        snapshot=lambda: np.zeros(0, dtype=np.float32))
    monkeypatch.setattr(flow, "foreground_window", lambda: "ed")
    monkeypatch.setattr("pyperclip.paste", lambda: "")
    monkeypatch.setattr("pyperclip.copy", lambda _t: None)

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"sid": "s1", "committed": "", "text": "final"}

    monkeypatch.setattr("requests.post", lambda url, **k: _Resp())
    return flow, f


def _started_client(monkeypatch):
    flow, f = _client_flow(monkeypatch)
    parked = threading.Event()
    _hook_idle(monkeypatch, flow, 0.35, parked)
    streamer = flow.ClientStreamer(f)
    streamer.start()
    assert streamer.sid == "s1" and streamer._thread is not None
    assert parked.wait(1.0), "idle wait never started"
    return streamer


def test_client_streamer_halt_cuts_the_idle_wait(monkeypatch):
    streamer = _started_client(monkeypatch)
    streamer.halt()
    t0 = time.perf_counter()
    text, samples = streamer.stop()
    elapsed = time.perf_counter() - t0
    alive = streamer._thread.is_alive()
    assert elapsed < 0.1 and not alive, (elapsed, alive)
    assert (text, samples) == ("", 0)


def test_client_streamer_finish_cuts_the_idle_wait(monkeypatch):
    streamer = _started_client(monkeypatch)
    t0 = time.perf_counter()
    text, typed = streamer.finish()
    elapsed = time.perf_counter() - t0
    alive = streamer._thread.is_alive()
    assert elapsed < 0.1 and not alive, (elapsed, alive)
    assert (text, typed) == ("final", 0)


def test_halt_during_decode_joins_and_commits_nothing(monkeypatch):
    flow, f = _stream_flow(monkeypatch, 0.01)
    entered = threading.Event()
    release = threading.Event()
    left = threading.Event()

    def decode(self, audio, **k):
        entered.set()
        release.wait(3)
        left.set()
        return _agreed_words()

    monkeypatch.setattr(flow.Flow, "decode", decode)
    worker = flow.StreamWorker(f)
    worker._prev = ["hello", "there"]
    worker.start()
    assert entered.wait(2.0), "decode never started"
    worker.halt()
    done = threading.Event()
    out = {}

    def join():
        t0 = time.perf_counter()
        out["ret"] = worker.stop()
        out["dt"] = time.perf_counter() - t0
        done.set()

    threading.Thread(target=join, daemon=True).start()
    time.sleep(0.2)
    assert not done.is_set(), "stop() returned while decode was still blocked"
    assert not left.is_set()
    assert worker.committed_text == ""
    assert worker.committed_samples == 0
    release.set()
    assert done.wait(1.0)
    assert left.is_set()
    assert 0.15 <= out["dt"] < 1.0
    assert not worker._thread.is_alive()
    assert out["ret"] == ("", 0)
    assert worker.committed_text == ""
    assert worker.committed_samples == 0
