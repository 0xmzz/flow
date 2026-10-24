"""Hands-free latch, scratch-that, rewrite selection, overlay hints.

No GPU, no network, no real clipboard. The latch machine is pure; the Flow
tests drive it through the same hooks the ctrl loop uses.
"""

import queue
import threading
import time
import types

import pytest


def _flow(monkeypatch, recording=False):
    import flow
    f = flow.Flow.__new__(flow.Flow)
    f.cfg = {
        "hotkey": "right ctrl",
        "command_hotkey": "right alt",
        "live_hotkey": "f9",
        "hermes": {"enabled": False, "hotkey": "right shift"},
        "max_record_s": 600,
        "max_latched_s": 7200,
        "mode": "full",
        "streaming": {"enabled": False},
        "mute_output_while_recording": False,
        "sample_rate": 16000,
        "min_duration_s": 0.6,
        "paste_mode": "clipboard",
        "server": {"token": "t"},
        "overlay": {"enabled": True, "hints": True},
    }
    f._pressed_keys = set()
    f._active_ptt_key = None
    f._key_lock = threading.Lock()
    f._key_seen_at = {}
    f._os_up_streak = 0
    f._record_started_at = None
    f._ctrl_q = queue.Queue()
    f._ctrl_stop = False
    f.recording = recording
    f.active_mode = "dictate"
    f.summary_take = False
    f.rewrite_take = False
    f.latched = False
    f._latch = dict(flow._LATCH_IDLE)
    f._alt_down = set()
    f._alt_held = False
    f._latched_shown = None
    f._ptt_times = __import__("collections").deque()
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
    f.recorder = types.SimpleNamespace(
        start=lambda: None, stop=lambda: b"", level=0.0)
    monkeypatch.setattr(flow, "stop_speech", lambda: None)
    monkeypatch.setattr(flow.Flow, "set_icon", lambda self, *a, **k: None)
    return f


def _ev(name, kind):
    import flow
    return types.SimpleNamespace(name=name, event_type=kind)


def _idle():
    import flow
    return dict(flow._LATCH_IDLE)


# --- double-tap latch -------------------------------------------------------

def test_latch_double_tap_then_a_later_press_stops():
    import flow
    st = _idle()
    st, act = flow.latch_step(st, "down", 10.0)
    assert act == "start" and st["phase"] == "holding"
    st, act = flow.latch_step(st, "up", 10.2)          # tap, under 0.35s
    assert act is None and st["phase"] == "pending"
    st, act = flow.latch_step(st, "down", 10.5)        # 0.3s after the release
    assert act == "latch" and st["phase"] == "latched"
    st, act = flow.latch_step(st, "up", 10.6)          # letting go does not stop
    assert act is None and st["phase"] == "latched"
    st, act = flow.latch_step(st, "down", 20.0)        # single tap while latched
    assert act == "stop" and st["phase"] == "idle"


def test_short_tap_without_a_second_press_releases():
    """A <0.35s tap that nobody follows still ends the take."""
    import flow
    st = _idle()
    st, _ = flow.latch_step(st, "down", 1.0)
    st, act = flow.latch_step(st, "up", 1.2)
    assert act is None and st["phase"] == "pending"
    st, act = flow.latch_step(st, "tick", 1.2 + flow._LATCH_GAP_S - 0.05)
    assert act is None
    st, act = flow.latch_step(st, "tick", 1.2 + flow._LATCH_GAP_S)
    assert act == "release" and st["phase"] == "idle"


def test_hold_releases_immediately_and_borderline_press_does_not_arm():
    import flow
    st, _ = flow.latch_step(_idle(), "down", 0.0)
    st, act = flow.latch_step(st, "up", flow._LATCH_MAX_PRESS_S)
    assert act == "release"
    st, _ = flow.latch_step(_idle(), "down", 0.0)
    st, act = flow.latch_step(st, "up", 1.2)
    assert act == "release" and st["phase"] == "idle"


def test_second_press_just_outside_the_gap_restarts():
    import flow
    st, _ = flow.latch_step(_idle(), "down", 0.0)
    st, _ = flow.latch_step(st, "up", 0.1)
    st, act = flow.latch_step(st, "down", 0.1 + flow._LATCH_GAP_S + 0.01)
    assert act == "restart" and st["phase"] == "holding"


def test_single_tap_stops_a_latched_take(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=True)
    f.latched = True
    f._latch = {"phase": "latched", "down_at": 1.0, "up_at": None}
    f._record_started_at = 1.0
    released, pressed = [], []
    monkeypatch.setattr(f, "on_release", lambda: released.append("up"))
    monkeypatch.setattr(f, "on_press", lambda mode: pressed.append(mode))
    f._latch_event("down", 5.0)
    assert released == ["up"]
    assert pressed == []
    assert f.latched is False


def test_short_tap_on_the_flow_still_releases(monkeypatch):
    f = _flow(monkeypatch)
    f._latch_event("down", 0.0)
    assert f.recording is True
    f._latch_event("up", 0.2)
    assert f.recording is True          # waiting to see if it's a double tap
    f._latch_event("tick", 0.2 + 0.4)
    assert f.recording is False
    assert f._job_q.get_nowait()[0] == "dictate"


def test_watchdog_ignores_a_latched_take(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=True)
    f._pressed_keys.add("right ctrl")
    f._active_ptt_key = "right ctrl"
    f.latched = True
    monkeypatch.setattr(flow, "_os_key_down", lambda vk: False)
    monkeypatch.setattr(f, "on_release", lambda: (_ for _ in ()).throw(
        AssertionError("a latched take stays open while the key is up")))
    f._poll_key_watchdog()
    f._poll_key_watchdog()
    assert f.recording is True
    assert f._active_ptt_key == "right ctrl"


def test_latched_cap_is_two_hours_and_the_hold_cap_is_unchanged(monkeypatch):
    import flow
    assert flow.DEFAULTS["max_latched_s"] == 7200
    assert flow.DEFAULTS["max_record_s"] == 600
    f = _flow(monkeypatch, recording=True)
    f._record_started_at = 0.0
    released = []
    monkeypatch.setattr(f, "on_release", lambda: released.append(1))
    monkeypatch.setattr(flow.time, "monotonic", lambda: 599.0)
    f._poll_record_cap()
    assert released == []
    monkeypatch.setattr(flow.time, "monotonic", lambda: 600.0)
    f._poll_record_cap()
    assert released == [1]               # a held key still stops at max_record_s
    released.clear()
    f.latched = True
    monkeypatch.setattr(flow.time, "monotonic", lambda: 601.0)
    f._poll_record_cap()                 # 601s is nothing on a 2h latch
    assert released == []
    monkeypatch.setattr(flow.time, "monotonic", lambda: 7200.0)
    f._poll_record_cap()
    assert released == [1]


def test_shift_during_a_latched_take_still_arms_summary(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=True)
    f.latched = True
    f._on_key(_ev("right shift", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.get_nowait() == ("summary", None)


# --- recorder trim ----------------------------------------------------------

def _recorder(monkeypatch):
    import flow
    monkeypatch.setattr(flow.Recorder, "_open", lambda self, _reinit=True: None)
    return flow.Recorder(16000, pre_roll_s=0)


def test_discard_prefix_keeps_snapshot_to_the_uncommitted_tail(monkeypatch):
    import numpy as np
    rec = _recorder(monkeypatch)
    rec._recording = True
    for i in range(100):
        rec._chunks.append(np.full((1600, 1), float(i), dtype=np.float32))
    assert len(rec.snapshot()) == 1600 * 100
    dropped = rec.discard_prefix(1600 * 90)
    assert dropped == 1600 * 90
    assert len(rec.snapshot()) == 1600 * 10
    assert sum(int(c.shape[0]) for c in rec._chunks) == 1600 * 10
    for _ in range(5):
        rec._on_audio(np.ones((1600, 1), dtype=np.float32))
    # New audio appends. The 90 dropped chunks are gone, not concatenated.
    assert len(rec.snapshot()) == 1600 * 15
    assert sum(int(c.shape[0]) for c in rec._chunks) == 1600 * 15


def test_discard_prefix_splits_a_straddling_chunk_and_stop_counts_the_take(monkeypatch):
    import numpy as np
    rec = _recorder(monkeypatch)
    rec._chunks = [np.arange(10, dtype=np.float32).reshape(10, 1)]
    assert rec.discard_prefix(4) == 4
    assert list(rec.snapshot()) == [4, 5, 6, 7, 8, 9]
    audio = rec.stop()
    assert list(audio) == [4, 5, 6, 7, 8, 9]
    assert rec.last_discarded == 4
    assert rec.last_full_samples == 10


def test_streamer_drops_audio_only_after_it_is_stored(monkeypatch):
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    worker = flow.StreamWorker(f, live=True)
    buf = {"n": np.ones(5000, dtype=np.float32)}

    def discard(n):
        n = min(int(n), len(buf["n"]))
        buf["n"] = buf["n"][n:]
        return n

    worker.source = types.SimpleNamespace(
        snapshot=lambda: buf["n"], discard_prefix=discard)
    worker.committed_text = "hello there"
    worker.committed_samples = 1000
    worker.typed_len = 0                 # live, not on screen yet
    worker._drop_committed_audio()
    assert worker.committed_samples == 1000
    assert len(buf["n"]) == 5000
    worker.typed_len = len(worker.committed_text)
    worker._drop_committed_audio()
    assert worker.committed_samples == 0
    assert len(buf["n"]) == 4000


# --- scratch that -----------------------------------------------------------

@pytest.mark.parametrize("src, expected", [
    ("Hello there. Scratch that.", ""),
    ("Keep this. Drop me. Scratch that.", "Keep this."),
    ("Keep this. Drop me. Scratch that. And this stays.",
     "Keep this. And this stays."),
    ("One. Scratch that. Two. Scratch that. C stays.", "C stays."),
    ("Scratch that.", ""),
    ("scratch that!", ""),
    ("Scratch that. Keep this.", "Keep this."),
])
def test_apply_voice_edits(src, expected):
    import flow
    assert flow.apply_voice_edits(src) == expected


def test_voice_edits_leave_unrelated_text_unchanged_and_honour_the_phrase_list():
    import flow
    assert flow.apply_voice_edits("Hello there.") == "Hello there."
    cfg = {"voice_edits": {"scratch": ["erase that"]}}
    assert flow.apply_voice_edits("Hello. Erase that.", cfg) == ""
    assert flow.apply_voice_edits("Hello. Scratch that.", cfg) == "Hello. Scratch that."
    assert flow.DEFAULTS["voice_edits"]["scratch"] == ["scratch that"]


def test_retract_erases_only_the_typed_sentence_a_scratch_removes():
    import flow
    old = "Hello there. Scratch that."
    spans = [(0, len("Hello there."), "Hello there.")]
    erase, retype, new_spans, new_len = flow.plan_retract(
        spans, old, "", len("Hello there."))
    assert erase == "Hello there."
    assert retype == ""
    assert new_len == 0
    assert new_spans == []


def test_retract_puts_back_a_typed_suffix_the_scratch_did_not_remove():
    import flow
    old = "Keep. Drop this. Scratch that. Stay."
    new = flow.apply_voice_edits(old)
    assert new == "Keep. Stay."
    spans = [(0, len(old), old)]
    erase, retype, spans_out, new_len = flow.plan_retract(
        spans, old, new, len(old))
    # The space after "Keep." stays in the kept prefix; only "Stay." is retyped.
    assert erase.endswith("Stay.")
    assert retype == "Stay."
    assert "".join(p for _s, _e, p in spans_out) == new
    assert new_len == len(new)


# --- rewrite selection ------------------------------------------------------

def test_alt_during_dictation_enqueues_rewrite_and_not_a_command(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=True)
    for name in ("left alt", "right alt", "alt"):
        f.rewrite_take = False
        f.summary_take = False
        f._on_key(_ev(name, flow.keyboard.KEY_DOWN))
        assert f._ctrl_q.get_nowait() == ("rewrite", None)
        assert f._ctrl_q.empty()


def test_right_alt_alone_is_still_a_command(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=False)
    f._on_key(_ev("right alt", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.get_nowait() == ("down", "command")


def test_ctrl_joining_an_alt_take_enqueues_rewrite(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=True)
    f.active_mode = "command"
    f._alt_held = True
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.get_nowait() == ("rewrite", None)


def test_ctrl_loop_routes_rewrite_and_stops_live_typing(monkeypatch):
    f = _flow(monkeypatch, recording=True)
    f.streamer = types.SimpleNamespace(live=True)
    f._ctrl_q.put(("rewrite", None))
    t = threading.Thread(target=f._ctrl_loop, daemon=True)
    t.start()
    deadline = time.time() + 2.0
    while not f.rewrite_take and time.time() < deadline:
        time.sleep(0.02)
    f._ctrl_stop = True
    f._ctrl_q.put(("toggle", None))
    t.join(1.0)
    assert f.rewrite_take is True
    assert f.streamer.live is False
    assert not t.is_alive()


def test_alt_already_held_opens_the_take_in_rewrite(monkeypatch):
    f = _flow(monkeypatch)
    f._alt_held = True
    f.on_press("dictate")
    assert f.recording is True
    assert f.rewrite_take is True


def test_rewrite_prompt_is_instruction_then_text(monkeypatch):
    import flow
    seen = {}

    def fake(text, cfg, system=None):
        seen["text"] = text
        seen["system"] = system
        return "rewritten"

    monkeypatch.setattr(flow, "llm_summarize", fake)
    assert flow.llm_rewrite("be brief", "Hello there", {}) == "rewritten"
    assert seen["system"] is flow.REWRITE_PROMPT
    assert seen["text"] == "Instruction: be brief\n\nText:\nHello there"


def test_summary_call_forwards_the_system_prompt(monkeypatch):
    import flow
    sent = {}

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "ok"}

    def post(url, timeout=None, headers=None, json=None):
        sent["url"] = url
        sent["json"] = json
        sent["headers"] = headers
        return Resp()

    monkeypatch.setattr(flow.requests, "post", post)
    out = flow._summary_call(
        {"provider": "flow-node", "model": "m", "url": "http://example.test/summarize"},
        "body", {"server": {"token": "tok"}}, system="SYS")
    assert out == "ok"
    assert sent["json"] == {"model": "m", "system": "SYS", "text": "body"}
    assert sent["headers"]["Authorization"] == "Bearer tok"
    assert "example.test" in sent["url"]


def test_rewrite_with_no_selection_pastes_nothing(monkeypatch):
    import flow
    f = _flow(monkeypatch)
    said, pasted, restored = [], [], []
    f.overlay = types.SimpleNamespace(set=lambda state: said.append(state))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "_restore_clip",
                        lambda saved, delay=0.0: restored.append((saved, delay)))
    monkeypatch.setattr(flow, "capture_selection", lambda: (None, "previous"))
    monkeypatch.setattr(flow, "llm_rewrite", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no selection must not call the model")))
    assert f._rewrite_selection("make it shorter") is None
    assert pasted == []
    assert ("say", "Select text first") in said
    assert restored == [("previous", 0.0)]


def test_rewrite_llm_failure_leaves_the_selection_and_pastes_nothing(monkeypatch):
    import flow
    f = _flow(monkeypatch)
    said, pasted, restored = [], [], []
    f.overlay = types.SimpleNamespace(set=lambda state: said.append(state))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "_restore_clip",
                        lambda saved, delay=0.0: restored.append((saved, delay)))
    monkeypatch.setattr(flow, "capture_selection",
                        lambda: ("Selected words", "previous"))
    monkeypatch.setattr(flow, "llm_rewrite", lambda *a, **k: None)
    assert f._rewrite_selection("make it shorter") is None
    assert pasted == []
    assert restored == [("previous", 0.0)]
    assert any(s[0] == "say" and "unchanged" in s[1] for s in said)


def test_rewrite_success_pastes_the_model_text_and_restores_later(monkeypatch):
    import flow
    f = _flow(monkeypatch)
    pasted, restored = [], []
    f.overlay = types.SimpleNamespace(set=lambda state: None)
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "_restore_clip",
                        lambda saved, delay=0.0: restored.append((saved, delay)))
    monkeypatch.setattr(flow, "capture_selection",
                        lambda: ("Selected words", "previous"))

    def fake(instruction, selection, cfg):
        assert instruction == "be brief"
        assert selection == "Selected words"
        return "Brief."

    monkeypatch.setattr(flow, "llm_rewrite", fake)
    assert f._rewrite_selection("be brief") == "Brief."
    assert pasted == [("Brief.", "clipboard")]
    assert restored == [("previous", 0.6)]


# --- overlay hints ----------------------------------------------------------

def test_hints_default_on_and_only_the_plain_recording_shows_them():
    import flow
    assert flow.DEFAULTS["overlay"]["hints"] is True
    assert flow._overlay_flags({"overlay": True}) == (True, True)
    assert flow._overlay_flags(
        {"overlay": {"enabled": True, "hints": False}}) == (True, False)
    status, ctx, hint = flow.rec_overlay(None, "→ Notepad", True)
    assert status == "Listening"
    assert ctx == "→ Notepad"
    assert hint == flow._HINT_LINE
    assert "scratch that" in hint
    assert flow.rec_overlay(None, "→ Notepad", False)[2] == ""
    assert flow.rec_overlay("summary", "", True)[2] == ""
    assert flow.rec_overlay("rewrite", "", True) == (
        "Listening · rewrite selection", "", "")
    assert flow.rec_overlay("handsfree", "01:05", True) == (
        "Hands-free · tap right ctrl to stop", "01:05", "")


def test_hint_line_truncates_instead_of_wrapping():
    import flow
    text = flow._HINT_LINE
    assert flow.clip_line(text, len(text)) == text
    clipped = flow.clip_line(text, 16)
    assert len(clipped) == 16
    assert clipped.endswith("…")
    assert "\n" not in clipped
