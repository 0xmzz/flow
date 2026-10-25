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
        # Product default is off. These tests opt in.
        "rewrite": {"enabled": True},
    }
    f._pressed_keys = set()
    f._active_ptt_key = None
    f._take_key = None
    f._owner_up_queued = None
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
    """Replaces the deferred seal/pending/second machine.

    The first release is an ordinary take. A second press inside the gap is
    only a candidate; a short release of that press converts it to a latch.
    """
    import flow
    st = _idle()
    st, act = flow.latch_step(st, "down", 10.0)
    assert act == "start" and st["phase"] == "holding"
    assert st["candidate"] is False
    st, act = flow.latch_step(st, "up", 10.2)          # tap, under 0.35s
    assert act == "release" and st["phase"] == "idle"
    assert st["prev_press_s"] == pytest.approx(0.2)
    st, act = flow.latch_step(st, "down", 10.5)        # 0.3s after the release
    assert act == "start" and st["phase"] == "holding"
    assert st["candidate"] is True
    st, act = flow.latch_step(st, "up", 10.6)          # short second release latches
    assert act == "latch" and st["phase"] == "latched"
    st, act = flow.latch_step(st, "down", 20.0)        # single tap while latched
    assert act == "stop" and st["phase"] == "idle"


def test_short_tap_without_a_second_press_releases():
    """A <0.35s tap ends at the key-up. Nothing is deferred across a tick."""
    import flow
    st = _idle()
    st, _ = flow.latch_step(st, "down", 1.0)
    st, act = flow.latch_step(st, "up", 1.2)
    assert act == "release" and st["phase"] == "idle"
    st, act = flow.latch_step(st, "tick", 1.2 + flow._LATCH_GAP_S)
    assert act is None and st["phase"] == "idle"


def test_hold_releases_immediately_and_borderline_press_does_not_arm():
    import flow
    st, _ = flow.latch_step(_idle(), "down", 0.0)
    st, act = flow.latch_step(st, "up", flow._LATCH_MAX_PRESS_S)
    assert act == "release"
    st, _ = flow.latch_step(_idle(), "down", 0.0)
    st, act = flow.latch_step(st, "up", 1.2)
    assert act == "release" and st["phase"] == "idle"


def test_second_press_just_outside_the_gap_restarts():
    """Replaces the old `restart` action. Outside the gap this is a plain start."""
    import flow
    st, _ = flow.latch_step(_idle(), "down", 0.0)
    st, _ = flow.latch_step(st, "up", 0.1)
    st, act = flow.latch_step(st, "down", 0.1 + flow._LATCH_GAP_S + 0.01)
    assert act == "start" and st["phase"] == "holding"
    assert st["candidate"] is False


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
    """Replaces the deferred tick. The key-up itself ends the take and queues it."""
    f = _flow(monkeypatch)
    f._latch_event("down", 0.0)
    assert f.recording is True
    f._latch_event("up", 0.2)
    assert f.recording is False
    assert f._job_q.get_nowait()[0] == "dictate"
    f._latch_event("tick", 0.2 + 0.4)
    assert f._job_q.empty()


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
    assert f.recording is True
    assert f._active_ptt_key == "right ctrl"   # one poll is not enough
    f._poll_key_watchdog()
    assert f.recording is True
    assert f._active_ptt_key is None           # bookkeeping only; the take stays
    assert "right ctrl" not in f._pressed_keys


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


def test_scratch_on_screen_keeps_the_typed_prefix_byte_for_byte():
    """A sentence already live-typed is not backspaced. The phrase is dropped."""
    import flow
    old = "Hello there. Scratch that."
    typed = len("Hello there.")
    text, kept = flow.voice_edit_keeping_screen(old, typed)
    assert text.startswith(old[:typed])
    assert kept == typed
    assert "Scratch that" not in text
    assert not hasattr(flow, "plan_retract")


def test_scratch_past_the_cursor_still_removes_the_untyped_sentence():
    import flow
    old = "Keep. Drop this. Scratch that. Stay."
    assert flow.apply_voice_edits(old) == "Keep. Stay."
    # Nothing is on screen yet, so the whole edit applies.
    text, kept = flow.voice_edit_keeping_screen(old, 0)
    assert text == "Keep. Stay."
    assert kept == 0


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
    assert f._ctrl_q.get_nowait() == ("down", "command", "right alt")


def test_ctrl_joining_an_alt_take_does_not_convert_it(monkeypatch):
    """Alt-first stays a command. Ctrl joining it is not a rewrite chord."""
    import flow
    f = _flow(monkeypatch, recording=True)
    f.active_mode = "command"
    f._active_ptt_key = "right alt"
    f._pressed_keys.add("right alt")
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.empty()
    assert f._active_ptt_key == "right alt"
    assert f.rewrite_take is False


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


def test_alt_already_held_does_not_open_the_take_as_rewrite(monkeypatch):
    """Alt pressed first is not rewrite. Rewrite is alt down during the take."""
    f = _flow(monkeypatch)
    f._alt_held = True
    f.on_press("dictate")
    assert f.recording is True
    assert f.active_mode == "dictate"
    assert f.rewrite_take is False


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
                        lambda saved, delay=0.0, expect_gen=None: restored.append((saved, delay)))
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
                        lambda saved, delay=0.0, expect_gen=None: restored.append((saved, delay)))
    monkeypatch.setattr(flow, "capture_selection",
                        lambda: ("Selected words", "previous"))
    monkeypatch.setattr(flow, "llm_rewrite", lambda *a, **k: None)
    assert f._rewrite_selection("make it shorter") is None
    assert pasted == []
    assert restored == [("previous", 0.0)]
    assert any(s[0] == "say" and "unchanged" in s[1] for s in said)


def test_rewrite_success_copies_and_does_not_paste(monkeypatch):
    """Success leaves the rewrite on the clipboard. No ctrl+v, no restore."""
    import flow
    f = _flow(monkeypatch)
    said, pasted, restored, copied = [], [], [], []
    f.overlay = types.SimpleNamespace(set=lambda state: said.append(state))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "_restore_clip",
                        lambda saved, delay=0.0, expect_gen=None: restored.append((saved, delay)))
    monkeypatch.setattr(flow, "_copy_clipboard",
                        lambda text, expect_gen=None: copied.append(text) or True)
    monkeypatch.setattr(flow, "capture_selection",
                        lambda: ("Selected words", "previous"))
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 11)
    monkeypatch.setattr(flow, "focused_control_hwnd", lambda: 42)
    monkeypatch.setattr(flow, "_last_input_tick", lambda: 1000)
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    sent = []
    monkeypatch.setattr(flow.keyboard, "send", lambda *a, **k: sent.append(a))

    def fake(instruction, selection, cfg):
        assert instruction == "be brief"
        assert selection == "Selected words"
        return "Brief."

    monkeypatch.setattr(flow, "llm_rewrite", fake)
    assert f._rewrite_selection("be brief") == "Brief."
    assert pasted == []
    assert restored == []
    assert sent == []
    assert copied == ["Brief."]
    assert ("say", "Rewritten — press Ctrl+V to replace") in said


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
    assert "Alt rewrite" not in hint
    assert flow.DEFAULTS["rewrite"] == {"enabled": False}
    enabled_hint = flow.overlay_hint_line({"rewrite": {"enabled": True}})
    assert "Alt rewrite" in enabled_hint
    assert flow.rec_overlay(None, "→ Notepad", True, enabled_hint)[2] == enabled_hint
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


def test_handsfree_take_commits_without_live_typing(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=True)
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: "")
    worker = flow.StreamWorker(f, live=True)
    worker._active = True
    f.latched = True
    assert worker._should_live_type() is False
    f.latched = False
    f.rewrite_take = True
    assert worker._should_live_type() is False
    f.rewrite_take = False
    assert worker._should_live_type() is True
    made = []

    class _Fake:
        def __init__(self, flow_obj, live=False, source=None):
            made.append(live)

        def start(self):
            return None

    monkeypatch.setattr(flow, "StreamWorker", _Fake)
    f.cfg["streaming"]["enabled"] = True
    f.active_mode = "dictate"
    f._open_streamer(1, handsfree=True)
    assert made == [False]


def test_latched_overlay_context_includes_the_word_count(monkeypatch):
    import flow
    assert flow.format_latched_context(252, 380) == "04:12 · 380 words"
    assert flow.format_latched_context(61, 1) == "01:01 · 1 word"
    assert flow.format_latched_context(0, 0) == "00:00 · 0 words"
    f = _flow(monkeypatch, recording=True)
    f.latched = True
    f._record_started_at = 1000.0 - 252
    f.streamer = types.SimpleNamespace(committed_text=" ".join(["word"] * 380))
    shown = []
    f.overlay = types.SimpleNamespace(set=lambda state: shown.append(state))
    monkeypatch.setattr(flow.time, "monotonic", lambda: 1000.0)
    f._poll_latched_clock()
    assert shown == [("rec", "04:12 · 380 words", "handsfree")]


# --- Astra round-2 regressions ---------------------------------------------

def test_astra_1_typed_replacement_is_not_reinvented():
    """Finding 1: a live-typed expansion stays. Scratch does not retype it."""
    import flow
    screen = "by the way."
    text = "by the way. Scratch that. Leave this."
    out, kept = flow.voice_edit_keeping_screen(text, len(screen))
    assert out[:len(screen)] == screen
    assert kept == len(screen)
    assert "Scratch that" not in out
    assert "Leave this." in out
    assert not hasattr(flow, "plan_retract")
    assert not hasattr(flow, "retract_toward")


def test_astra_2_stale_trim_cannot_rebase_or_eat_the_next_take(monkeypatch):
    """Finding 2: a sealed streamer cannot trim the buffer it no longer owns.

    halt() does not wait. Sealing copies the audio and the commit offsets
    together, then advances the generation. A later trim must not rebase
    those offsets onto a different buffer, and must not discard the next take.
    """
    import numpy as np
    import flow
    rec = _recorder(monkeypatch)
    rec._stream = types.SimpleNamespace(active=True)
    gen = rec.start()
    rec._chunks = [np.arange(1000, dtype=np.float32).reshape(-1, 1)]
    streamer = types.SimpleNamespace(
        committed_text="hello there", committed_samples=400, _frozen=None)
    audio, frozen = rec.capture(streamer)
    assert frozen == ("hello there", 400)
    assert streamer._frozen == frozen
    assert len(audio) == 1000
    assert rec.discard_prefix(400, gen) == 0
    assert streamer.committed_samples == 400
    gen2 = rec.start()
    assert gen2 != gen
    rec._chunks = [np.ones((800, 1), dtype=np.float32)]
    assert len(rec.snapshot(gen)) == 0
    assert len(rec.snapshot(gen2)) == 800
    assert rec.discard_prefix(800, gen) == 0
    assert len(rec.snapshot(gen2)) == 800
    assert float(rec.snapshot(gen2)[0]) == 1.0
    f = _flow(monkeypatch)
    worker = flow.StreamWorker(f, live=False)
    worker.source = rec
    worker._gen = gen
    worker._frozen = None
    worker.committed_text = "hello there"
    worker.committed_samples = 400
    worker.typed_len = len(worker.committed_text)
    worker._drop_committed_audio()
    assert worker.committed_samples == 400
    assert len(rec.snapshot(gen2)) == 800


def test_astra_3_client_upload_does_not_trim_local_audio(monkeypatch):
    """Finding 3: client mode never discards samples, and finish() uploads
    the sealed take rather than whatever the recorder has started since."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["mode"] = "client"
    f.cfg["server_url"] = "http://lane.test"
    f.resolve_lane = lambda *a, **k: "http://lane.test"
    trimmed = []
    current = {"audio": np.arange(8000, dtype=np.float32)}

    class _Rec:
        _gen = 1

        def snapshot(self, gen=None):
            if gen is not None and gen != self._gen:
                return np.zeros(0, dtype=np.float32)
            return current["audio"]

        def discard_prefix(self, n, gen=None):
            trimmed.append((n, gen))
            return 0

    f.recorder = _Rec()
    posts = []

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"committed": "", "text": "said"}

    streamer = flow.ClientStreamer(f)
    streamer.sid = "s"
    streamer._gen = 1
    streamer.sent = 0

    def post(url, data=None, **k):
        posts.append((url, None if data is None else bytes(data)))
        streamer._active = False
        return _Resp()

    monkeypatch.setattr("requests.post", post)
    streamer._active = True
    streamer._wake.set()
    t = threading.Thread(target=streamer._loop, daemon=True)
    t.start()
    t.join(2.0)
    assert not t.is_alive()
    assert trimmed == []
    assert streamer.sent == 8000
    assert posts and "/audio" in posts[0][0]
    assert posts[0][1] is not None and len(posts[0][1]) == 8000 * 2

    current["audio"] = np.ones(20000, dtype=np.float32)
    f.recorder._gen = 2
    sealed = np.arange(4000, dtype=np.float32)
    streamer.sent = 1000
    streamer._thread = None
    before = len(posts)
    text, _typed = streamer.finish()
    assert text == "said"
    audio_posts = [p for p in posts[before:] if "/audio" in p[0]]
    assert len(audio_posts) == 1
    # 9762188 finish() reads the live recorder, not a sealed argument.
    assert len(audio_posts[0][1]) == 19000 * 2
    assert trimmed == []

    seen = {}

    def fake_finish():
        seen["called"] = True
        return None, 0

    def transcribe(audio, route, **k):
        seen["fallback"] = audio
        return "from fallback"

    streamer.finish = fake_finish
    streamer.sid = "s"
    f.cfg["min_duration_s"] = 0
    monkeypatch.setattr(f, "mic_is_deaf", lambda audio: False)
    monkeypatch.setattr(f, "client_transcribe", transcribe)
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: None)
    f.process(sealed, streamer=streamer)
    assert seen["called"] is True
    assert seen["fallback"] is sealed


def test_astra_4_rewrite_and_summary_refuse_the_wrong_window(monkeypatch):
    import flow
    f = _flow(monkeypatch)
    said, pasted, copied = [], [], []
    f.overlay = types.SimpleNamespace(set=lambda state: said.append(state))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "_restore_clip", lambda *a, **k: None)
    monkeypatch.setattr(flow, "capture_selection",
                        lambda: ("Selected words", "previous", None))
    monkeypatch.setattr(flow, "llm_rewrite", lambda *a, **k: "Brief.")
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 22)
    monkeypatch.setattr(flow, "_copy_clipboard",
                        lambda text: copied.append(text) or True)
    assert f._rewrite_selection("be brief", take_hwnd=11) == "Brief."
    assert pasted == []
    assert copied == ["Brief."]
    assert ("say", "Rewritten — press Ctrl+V to replace") in said

    # An untyped summary still refuses the wrong window, and never erases.
    assert not hasattr(flow.Flow, "_erase_typed")
    monkeypatch.setattr(flow, "_last_input_tick", lambda: None)
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: False)
    assert f._summary_target_ok(11, None) == ("copy", "Summary copied")
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 22)
    assert f._summary_target_ok(11, None) == (
        "copy", "Summary copied — window changed")
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 11)
    assert f._summary_target_ok(11, None) == "paste"


def test_astra_5_scratch_that_never_sends_backspace(monkeypatch):
    import flow
    sent = []
    monkeypatch.setattr(flow.keyboard, "send", lambda *a, **k: sent.append(a))
    f = _flow(monkeypatch)
    typed = len("Hello there.")
    text, kept = f._apply_voice_edit("Hello there. Scratch that.", typed)
    assert sent == []
    assert kept == typed
    assert text.startswith("Hello there.")
    assert "Scratch that" not in text
    assert not hasattr(flow, "plan_retract")


def test_astra_6_client_mode_does_not_rewrite_or_live_type_it(monkeypatch):
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["mode"] = "client"
    f._alt_held = True
    f.on_press("dictate")
    assert f.recording is True
    assert f.rewrite_take is False
    f._enable_rewrite()
    assert f.rewrite_take is False
    sent = []
    monkeypatch.setattr(flow.keyboard, "send", lambda *a, **k: sent.append(a))
    monkeypatch.setattr(flow.pyperclip, "copy", lambda *a, **k: None)
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    streamer = flow.ClientStreamer(f)
    streamer.committed = "replacement"
    streamer.typed = 0
    # The 9762188 client typer does not consult rewrite or latch.
    f.rewrite_take = True
    f.latched = True
    streamer._type_delta()
    assert sent == [("ctrl+v",)]
    assert streamer.typed == len("replacement")
    f.rewrite_take = False
    f.latched = False
    f.cfg["min_duration_s"] = 0
    pasted = []
    monkeypatch.setattr(f, "mic_is_deaf", lambda audio: False)
    monkeypatch.setattr(f, "client_transcribe", lambda *a, **k: "dictated line")
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    rewritten = []
    monkeypatch.setattr(f, "_rewrite_selection",
                        lambda *a, **k: rewritten.append(a))
    f.process(np.ones(1600, dtype=np.float32), rewrite=True)
    assert rewritten == []
    assert pasted and pasted[0][0] == "dictated line"


def test_astra_7_latch_confirms_on_the_second_release_not_the_second_down(monkeypatch):
    """Replaces seal/second/restart. The second down only arms a candidate.

    A release of exactly 0.35s ends the take. A shorter release of that
    candidate converts it to a latch. A long second press is a normal take.
    """
    import numpy as np
    import flow
    st = _idle()
    st, _ = flow.latch_step(st, "down", 0.0)
    st, act = flow.latch_step(st, "up", 0.10)
    assert act == "release" and st["phase"] == "idle"
    st, act = flow.latch_step(st, "down", 0.10 + flow._LATCH_GAP_S)
    assert act == "start" and st["candidate"] is False  # gap of exactly 0.40s
    st = _idle()
    st, _ = flow.latch_step(st, "down", 0.0)
    st, _ = flow.latch_step(st, "up", 0.10)
    st, act = flow.latch_step(st, "down", 0.30)
    assert act == "start" and st["phase"] == "holding" and st["candidate"] is True
    assert act != "latch"
    # Same float as the threshold, measured from down_at 0, so the compare is exact.
    border = dict(st)
    border["down_at"] = 0.0
    held_out, act = flow.latch_step(border, "up", flow._LATCH_MAX_PRESS_S)
    assert act == "release" and held_out["phase"] == "idle"
    short = dict(st)
    short["down_at"] = 0.0
    latched, act = flow.latch_step(short, "up", 0.20)
    assert act == "latch" and latched["phase"] == "latched"

    f = _flow(monkeypatch)
    rec = _recorder(monkeypatch)
    rec._stream = types.SimpleNamespace(active=True)
    f.recorder = rec
    f._latch_event("down", 0.0)
    rec._on_audio(np.ones((15, 1), dtype=np.float32))
    f._latch_event("up", 0.10)
    assert f.recording is False
    f._latch_event("down", 0.20)
    assert f._latch["phase"] == "holding"
    assert f._latch["candidate"] is True
    assert f.latched is False
    assert f.recording is True
    rec._on_audio(np.full((12, 1), 4, dtype=np.float32))
    f._latch_event("up", 0.20 + 0.50)
    assert f.recording is False
    assert f.latched is False
    first = f._job_q.get_nowait()
    second = f._job_q.get_nowait()
    assert first[0] == second[0] == "dictate"
    assert np.all(np.asarray(first[1][0]) == 1)
    assert np.all(np.asarray(second[1][0]) == 4)
    assert f._job_q.empty()


def test_astra_8_alt_first_stays_a_command(monkeypatch):
    """Ctrl joining alt does not take ownership or arm rewrite."""
    import flow
    f = _flow(monkeypatch, recording=True)
    f.active_mode = "command"
    f._active_ptt_key = "right alt"
    f._pressed_keys.add("right alt")
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    assert f._active_ptt_key == "right alt"
    assert f._ctrl_q.empty()
    assert f.rewrite_take is False
    f._on_key(_ev("right alt", flow.keyboard.KEY_UP))
    assert f._ctrl_q.get_nowait() == ("up", "command", "right alt")
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_UP))
    assert f._ctrl_q.empty()
    assert "right ctrl" not in f._pressed_keys
    assert not hasattr(f, "_plain_release")


def test_astra_9_remote_session_ignores_and_clears_local_latch(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=False)
    f.latched = True
    f._latch = {"phase": "latched", "down_at": 1.0, "up_at": 1.2}
    f.active_mode = "dictate"
    assert f.remote_stop() is None
    assert f.latched is True
    assert f._latch["phase"] == "latched"
    f.recording = True
    f.active_mode = "remote"
    before = dict(f._latch)
    f._latch_event("down", 10.0)
    f._latch_event("up", 10.2)
    f._latch_event("down", 10.4)
    f._latch_event("up", 10.5)
    assert f._latch == before
    assert f.recording is True
    assert f.remote_cancel() is True
    assert f.latched is False
    assert f._latch["phase"] == "idle"
    assert f.recording is False
    f._latch_event("down", 30.0)
    assert f.recording is True
    assert f._latch["phase"] == "holding"


def test_astra_10_latched_watchdog_clears_a_stuck_key_without_stopping(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=True)
    f.latched = True
    f._latch = {"phase": "latched", "down_at": 1.0, "up_at": 1.2}
    f._pressed_keys.add("right ctrl")
    f._active_ptt_key = "right ctrl"
    clock = {"t": 1000.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])
    f._key_seen_at["right ctrl"] = 1000.0
    released = []
    monkeypatch.setattr(f, "on_release", lambda: released.append("up"))
    monkeypatch.setattr(flow, "_os_key_down", lambda vk: False)
    f._poll_key_watchdog()
    assert "right ctrl" in f._pressed_keys
    f._poll_key_watchdog()
    assert released == []
    assert f.recording is True
    assert f.latched is True
    assert f._active_ptt_key is None
    assert "right ctrl" not in f._pressed_keys
    clock["t"] = 1000.4
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.get_nowait() == ("down", "dictate", "right ctrl")
    f._latch_event("down", clock["t"])
    assert released == ["up"]
    assert f.latched is False


def test_astra_11_client_scratch_keeps_the_typed_prefix():
    import flow
    text = "Drop me. Scratch that. Keep this."
    typed = len("Drop me.")
    out, kept = flow.voice_edit_keeping_screen(text, typed)
    assert kept == typed
    assert out.startswith(text[:typed])
    assert "Scratch that" not in out[typed:]
    assert "Keep this." in out[typed:]


def test_astra_12_scratch_does_not_zero_the_typing_ledger(monkeypatch):
    import flow
    f = _flow(monkeypatch)
    worker = flow.StreamWorker(f, live=False)
    typed = len("Hello there.")
    worker.committed_text = "Hello there."
    worker.typed_len = typed
    worker.typed_chars = typed
    worker.source = types.SimpleNamespace(
        snapshot=lambda *a, **k: b"", discard_prefix=None, _buf_lock=None)
    worker._commit_phrase("Scratch that.", 100, [])
    assert worker.typed_len == typed
    assert worker.typed_chars == typed
    assert worker.committed_text.startswith("Hello there.")


def test_astra_13_sentinel_detects_no_selection_and_accepts_a_matching_clipboard(monkeypatch):
    import flow
    assert "VS Code" in (flow.capture_selection.__doc__ or "")
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    clip = {"v": "already there"}

    def paste():
        return clip["v"]

    def copy(text):
        clip["v"] = text

    monkeypatch.setattr(flow.pyperclip, "paste", paste)
    monkeypatch.setattr(flow.pyperclip, "copy", copy)
    monkeypatch.setattr(flow.keyboard, "send", lambda *a, **k: None)
    text, saved, problem = flow.capture_selection()
    assert text is None and problem == "nosel"
    assert saved == "already there"
    assert clip["v"].startswith("flow-sel-")

    clip["v"] = "already there"

    def send(keys):
        if keys == "ctrl+c":
            clip["v"] = "already there"

    monkeypatch.setattr(flow.keyboard, "send", send)
    text, saved, problem = flow.capture_selection()
    assert problem is None
    assert text == "already there"
    assert saved == "already there"


def test_astra_14_held_modifiers_block_copy_and_paste(monkeypatch):
    import flow
    sent = []
    monkeypatch.setattr(flow.keyboard, "send", lambda *a, **k: sent.append(a))
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(flow, "_os_key_down", lambda vk: True)
    ticks = {"n": 0}

    def mono():
        ticks["n"] += 1
        return ticks["n"] * 10.0

    monkeypatch.setattr(flow.time, "monotonic", mono)
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: "old")
    text, saved, problem = flow.capture_selection()
    assert (text, problem) == (None, "modifiers")
    assert saved == "old"
    assert sent == []

    f = _flow(monkeypatch)
    said, pasted, restored = [], [], []
    f.overlay = types.SimpleNamespace(set=lambda state: said.append(state))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "_restore_clip",
                        lambda saved, delay=0.0, expect_gen=None: restored.append(saved))
    calls = {"n": 0}

    def wait(*a, **k):
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr(flow, "wait_modifiers_up", wait)
    clip = {"v": "previous"}
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: clip["v"])

    def copy(text):
        clip["v"] = text

    monkeypatch.setattr(flow.pyperclip, "copy", copy)

    def send(keys):
        sent.append(keys)
        if keys == "ctrl+c":
            clip["v"] = "Selected words"

    monkeypatch.setattr(flow.keyboard, "send", send)
    monkeypatch.setattr(flow, "llm_rewrite", lambda *a, **k: "Brief.")
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 1)
    assert f._rewrite_selection("be brief", take_hwnd=1) is None
    assert pasted == []
    assert "ctrl+v" not in sent
    assert "ctrl+c" in sent
    assert ("say", "Let go of Ctrl, Alt and Shift — rewrite cancelled") in said
    assert restored[-1] == "previous"


def test_astra_15_short_tap_seals_audio_at_the_first_release(monkeypatch):
    """Replaces the sealed-gap take. The short tap's job is its own audio.

    A following short press inside the gap becomes hands-free and keeps
    recording, including the audio of that second press. Nothing is discarded
    to stitch the two gestures, and no second job is queued until the stop.
    """
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    rec = _recorder(monkeypatch)
    rec._stream = types.SimpleNamespace(active=True)
    f.recorder = rec
    f._latch_event("down", 0.0)
    rec._on_audio(np.ones((40, 1), dtype=np.float32))
    f._latch_event("up", 0.2)
    assert f.recording is False
    rec._on_audio(np.full((25, 1), 9, dtype=np.float32))
    kind, args = f._job_q.get_nowait()
    assert kind == "dictate"
    assert np.array_equal(np.asarray(args[0]), np.ones(40, dtype=np.float32))
    assert 9 not in np.asarray(args[0])
    assert f._job_q.empty()

    f2 = _flow(monkeypatch)
    rec2 = _recorder(monkeypatch)
    rec2._stream = types.SimpleNamespace(active=True)
    f2.recorder = rec2
    f2._latch_event("down", 0.0)
    rec2._on_audio(np.full((40, 1), 1, dtype=np.float32))
    f2._latch_event("up", 0.1)
    assert f2.recording is False
    assert f2._job_q.get_nowait()[0] == "dictate"
    f2._latch_event("down", 0.2)
    rec2._on_audio(np.full((10, 1), 8, dtype=np.float32))
    f2._latch_event("up", 0.3)
    assert f2.latched is True
    assert f2.recording is True
    assert f2._latch["phase"] == "latched"
    snap = rec2.snapshot()
    assert np.all(snap == 8)
    assert f2._job_q.empty()


@pytest.mark.parametrize("src, expected", [
    ("Keep. Drop. Scratch that and continue.", "Keep. And continue."),
    ("Scratch that and continue.", "And continue."),
    ("Keep. Please drop scratch that and go.", "Keep. And go."),
    ("Keep this. Drop me. Scratch that. And this stays.",
     "Keep this. And this stays."),
])
def test_astra_16_scratch_cuts_from_the_command_position(src, expected):
    import flow
    assert flow.apply_voice_edits(src) == expected


def _queued(f):
    items = []
    while True:
        try:
            items.append(f._ctrl_q.get_nowait())
        except queue.Empty:
            break
    for item in items:
        f._ctrl_q.put(item)
    return items


def _play_ctrl(f):
    """Control-thread order: the same branches _ctrl_loop runs, without the poll."""
    while True:
        try:
            item = f._ctrl_q.get_nowait()
        except queue.Empty:
            return
        action = item[0]
        mode = item[1] if len(item) > 1 else None
        key = item[2] if len(item) > 2 else None
        if action == "rewrite":
            f._enable_rewrite()
        elif action == "summary":
            f._enable_summary()
        elif action == "down":
            when = f._pop_ptt_time()
            if mode == "dictate" and f.cfg.get("mode") != "client":
                f._latch_event("down", when, key=key)
            elif key is None:
                f.on_press(mode)
            else:
                f.on_press(mode, key=key)
        elif action == "up":
            when = f._pop_ptt_time()
            f._handle_up(mode, when, key=key)
        else:
            f.on_release()


# --- Astra round-2b regressions ------------------------------------------------
# Rewritten in place (deferral removed, not deleted):
#   test_latch_double_tap_then_a_later_press_stops
#   test_short_tap_without_a_second_press_releases
#   test_second_press_just_outside_the_gap_restarts
#   test_short_tap_on_the_flow_still_releases
#   test_astra_7_latch_confirms_on_the_second_release_not_the_second_down
#   test_astra_15_short_tap_seals_audio_at_the_first_release
#   test_astra_4 summary half (no paste on a refused erase)
#   test_astra_9 (no sealed blob)


def test_astra_r2_1_trim_rebases_the_offset_inside_the_lock(monkeypatch):
    """Samples 0..9, offset 4. Capture runs as the trim lock drops.

    The frozen offset must already be 0, so the tail is [4..9], not [8, 9].
    """
    import numpy as np
    import flow
    rec = _recorder(monkeypatch)
    rec._recording = True
    rec._chunks = [np.arange(10, dtype=np.float32).reshape(-1, 1)]
    owner = types.SimpleNamespace(committed_samples=4, committed_text="", _frozen=None)
    captured = {}

    class _Hook:
        def __init__(self, inner):
            self._inner = inner
            self._fire = True

        def __enter__(self):
            self._inner.acquire()
            return self

        def __exit__(self, *exc):
            self._inner.release()
            if self._fire:
                self._fire = False
                audio, frozen = rec.capture(owner)
                captured["audio"] = np.array(audio, copy=True)
                captured["offset"] = frozen[1]
            return False

    rec._buf_lock = _Hook(rec._buf_lock)
    assert rec.discard_prefix(4, owner=owner) == 4
    audio = captured["audio"]
    offset = captured["offset"]
    assert np.array_equal(audio, np.arange(4, 10, dtype=np.float32))
    assert offset == 0
    assert np.array_equal(audio[offset:], np.arange(4, 10, dtype=np.float32))
    assert owner.committed_samples == 0


def test_astra_r2_2_rewrite_and_summary_verify_the_target(monkeypatch):
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    said, pasted, copied = [], [], []
    f.overlay = types.SimpleNamespace(set=lambda state: said.append(state))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "_restore_clip", lambda *a, **k: None)
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow, "_copy_clipboard", lambda text: copied.append(text) or True)
    monkeypatch.setattr(flow, "llm_rewrite", lambda *a, **k: "Brief.")
    hwnd = {"v": 11}
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: hwnd["v"])
    calls = {"n": 0}

    def capture():
        calls["n"] += 1
        return ("Selected words", "previous", None)

    monkeypatch.setattr(flow, "capture_selection", capture)

    def move(*a, **k):
        hwnd["v"] = 99
        return "Brief."

    monkeypatch.setattr(flow, "llm_rewrite", move)
    assert f._rewrite_selection("be brief", take_hwnd=11) == "Brief."
    assert calls["n"] == 1
    assert pasted == []
    assert copied == ["Brief."]
    assert ("say", "Rewritten — press Ctrl+V to replace") in said

    hwnd["v"] = 11
    said.clear()
    copied.clear()
    pasted.clear()
    seq = iter([
        ("Selected words", "previous", None),
        ("Different words", "previous", None),
    ])
    monkeypatch.setattr(flow, "capture_selection", lambda: next(seq))
    monkeypatch.setattr(flow, "llm_rewrite", lambda *a, **k: "Brief.")
    assert f._rewrite_selection("be brief", take_hwnd=11) == "Brief."
    assert pasted == []
    assert copied == ["Brief."]
    assert ("say", "Rewritten — press Ctrl+V to replace") in said

    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(f, "mic_is_deaf", lambda audio: False)
    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *a, **k: ("spoken words here", {}))
    monkeypatch.setattr(flow, "llm_summarize", lambda *a, **k: "SHORT")
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    monkeypatch.setattr(flow, "_last_input_tick", lambda: 5000)
    f.cfg["min_duration_s"] = 0
    streamer = types.SimpleNamespace(
        typed_len=4, typed_chars=4, stop=lambda: None,
        committed_text="", committed_samples=0,
        saved_clip=None, restore_clipboard=lambda: None)
    said.clear()
    pasted.clear()
    copied.clear()
    audio = np.ones(32, dtype=np.float32)
    f.process(audio, streamer=streamer, summary=True, take_hwnd=11, input_tick=1000)
    assert pasted == []
    assert "SHORT" in copied
    assert ("say", "Summary copied — paste to replace") in said

    said.clear()
    pasted.clear()
    copied.clear()
    hwnd["v"] = 99
    monkeypatch.setattr(flow, "_last_input_tick", lambda: None)
    f.process(audio, streamer=streamer, summary=True, take_hwnd=11, input_tick=None)
    assert pasted == []
    assert "SHORT" in copied
    assert ("say", "Summary copied — paste to replace") in said


def test_astra_r2_3_alt_ctrl_chord_converts_before_recording_starts(monkeypatch):
    """Removed by design: alt-first is a command, and rewrite does not stick."""
    import flow
    f = _flow(monkeypatch)
    f.recording = False
    f._on_key(_ev("right alt", flow.keyboard.KEY_DOWN))
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    assert f._active_ptt_key == "right alt"
    assert f.recording is False
    assert not hasattr(f, "_chord_pending")
    f._on_key(_ev("right alt", flow.keyboard.KEY_UP))
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_UP))
    items = _queued(f)
    assert items == [("down", "command", "right alt"), ("up", "command", "right alt")]
    _play_ctrl(f)
    assert f.recording is False
    kind, _args = f._job_q.get_nowait()
    assert kind == "command"

    fresh = _flow(monkeypatch)
    fresh._enable_rewrite()
    assert fresh.rewrite_take is False
    fresh.on_press("command")
    assert fresh.recording is True
    assert fresh.active_mode == "command"
    assert fresh.rewrite_take is False


def test_astra_r2_4_remote_stop_cannot_clear_a_local_take(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=True)
    f.active_mode = "remote"
    f._mic_held = True
    assert f._mic_lock.acquire(blocking=False)

    class _Gate:
        def __init__(self, inner):
            self.inner = inner
            self.paused = threading.Event()
            self.go = threading.Event()
            self._first = True
            self._mu = threading.Lock()

        def __enter__(self):
            self.inner.acquire()
            with self._mu:
                pause = self._first
                self._first = False
            if pause:
                self.paused.set()
                assert self.go.wait(3)
            return self

        def __exit__(self, *exc):
            self.inner.release()
            return False

    gate = _Gate(f._session_lock)
    f._session_lock = gate
    errors = []

    def stop():
        try:
            errors.append(f.remote_stop())
        except Exception as exc:
            errors.append(exc)

    remote = threading.Thread(target=stop)
    remote.start()
    assert gate.paused.wait(2)
    started = threading.Event()
    done = threading.Event()

    def local():
        started.set()
        f._latch_event("down", 1000.0)
        done.set()

    press = threading.Thread(target=local)
    press.start()
    assert started.wait(2)
    time.sleep(0.05)
    assert not done.is_set()
    gate.go.set()
    remote.join(3)
    press.join(3)
    assert not remote.is_alive() and not press.is_alive()
    assert errors == [""]
    assert f.recording is True
    assert f._latch["phase"] == "holding"
    assert f._latch_owner == "local"
    f._latch_event("up", 1001.0)
    assert f.recording is False
    assert f._mic_held is False
    assert f._job_q.get_nowait()[0] == "dictate"

    held = _flow(monkeypatch, recording=True)
    held._latch_owner = "local"
    held._latch = dict(flow._LATCH_IDLE)
    held._latch["phase"] = "holding"
    held._clear_latch()
    assert held._latch["phase"] == "holding"
    assert held._latch_owner == "local"


def test_astra_r2_5_client_release_keeps_the_second_takes_audio(monkeypatch):
    """Removed by design: client mode has no latch and no hands-free discard."""
    import numpy as np
    import flow
    assert not hasattr(flow.Flow, "_enter_handsfree")
    f = _flow(monkeypatch)
    f.cfg["mode"] = "client"
    rec = _recorder(monkeypatch)
    rec._stream = types.SimpleNamespace(active=True)
    f.recorder = rec
    f._latch_event("down", 0.0)
    rec._on_audio(np.ones((100, 1), dtype=np.float32))
    f._latch_event("up", 0.2)
    assert f.recording is False
    assert f.latched is False
    kind, args = f._job_q.get_nowait()
    assert kind == "dictate"
    assert np.array_equal(np.asarray(args[0]), np.ones(100, dtype=np.float32))
    f._latch_event("down", 0.3)
    rec._on_audio(np.full((50, 1), 2, dtype=np.float32))
    f._latch_event("up", 0.5)
    assert f.recording is False
    kind, args = f._job_q.get_nowait()
    assert np.all(np.asarray(args[0]) == 2)
    assert len(args[0]) == 50


def test_astra_r2_6_a_short_ctrl_release_lets_the_next_mode_record(monkeypatch):
    """Removed by design: there is no sealed gap that swallows the next hold."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    rec = _recorder(monkeypatch)
    rec._stream = types.SimpleNamespace(active=True)
    f.recorder = rec
    f._latch_event("down", 0.0)
    rec._on_audio(np.ones((40, 1), dtype=np.float32))
    f._latch_event("up", 0.2)
    assert f.recording is False
    first = f._job_q.get_nowait()
    assert np.all(np.asarray(first[1][0]) == 1)
    f._on_key(_ev("right alt", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.get_nowait() == ("down", "command", "right alt")
    f.on_press("command")
    assert f.recording is True
    assert f.active_mode == "command"
    rec._on_audio(np.full((80, 1), 3, dtype=np.float32))
    f.on_release()
    assert f.recording is False
    second = f._job_q.get_nowait()
    assert second[0] == "command"
    assert len(second[1][0]) == 80
    assert np.all(np.asarray(second[1][0]) == 3)


def test_astra_r2_7_published_prefix_and_client_remainder_stay_intact(monkeypatch):
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    worker = flow.StreamWorker(f, live=False)
    worker._active = True
    worker.committed_text = "Drop me."
    worker.note_published(len("Drop me."))
    worker._commit_phrase("Scratch that.", 10, [])
    assert worker.committed_text.startswith("Drop me.")
    assert "Scratch that" not in worker.committed_text


def test_astra_r2_8_client_streamer_types_the_baseline_delta(monkeypatch):
    """ClientStreamer matches 9762188: a committed delta is pasted.

    halt() still stops the upload loop. The baseline typer does not also
    refuse `_type_delta` after halt. Server-mode StreamWorker does, via
    `_should_live_type`.
    """
    import flow
    f = _flow(monkeypatch)
    sent = []
    monkeypatch.setattr(flow.keyboard, "send", lambda *a, **k: sent.append(a))
    monkeypatch.setattr(flow.pyperclip, "copy", lambda *a, **k: None)
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    streamer = flow.ClientStreamer(f)
    streamer._active = False
    streamer.committed = "hello"
    streamer.typed = 0
    streamer._type_delta()
    assert sent == [("ctrl+v",)]
    assert streamer.typed == len("hello")


def test_astra_r2_9_retained_result_survives_streamer_restore(monkeypatch):
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    copied = []
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(flow.pyperclip, "copy", lambda text: copied.append(text))
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: "")
    monkeypatch.setattr(f, "mic_is_deaf", lambda audio: False)
    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *a, **k: ("be brief", {}))
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    hwnd = {"v": 11}
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: hwnd["v"])
    monkeypatch.setattr(flow, "capture_selection",
                        lambda: ("Selected words", "previous", None))

    def move(*a, **k):
        hwnd["v"] = 22
        return "BRIEF REWRITE"

    monkeypatch.setattr(flow, "llm_rewrite", move)
    said = []
    f.overlay = types.SimpleNamespace(set=lambda state: said.append(state))
    worker = flow.StreamWorker(f, live=False)
    worker.saved_clip = "ORIGINAL CLIPBOARD"
    worker.stop = lambda: None
    f.process(np.ones(16, dtype=np.float32), streamer=worker, rewrite=True,
              take_hwnd=11)
    assert ("say", "Rewritten — press Ctrl+V to replace") in said
    assert copied
    assert copied[-1] == "BRIEF REWRITE"
    assert "ORIGINAL CLIPBOARD" not in copied


def test_astra_r2_10_each_scratch_is_judged_against_the_prefix():
    import flow
    src = "Keep. Scratch that. Drop. Scratch that. Continue."
    out, kept = flow.voice_edit_keeping_screen(src, len("Keep."))
    assert out == "Keep. Continue."
    assert out.startswith("Keep.")
    assert kept == len("Keep.")


def test_astra_r2_11_vscode_copy_line_is_a_documented_limitation(monkeypatch):
    """Copy-line is not rejected. A one-line clipboard is a selection."""
    import flow
    assert not hasattr(flow, "_vscode_copy_line")
    doc = flow.capture_selection.__doc__ or ""
    assert "VS Code" in doc
    assert "known limitation" in doc
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 1)
    monkeypatch.setattr(flow, "focused_control_hwnd", lambda: 2)
    monkeypatch.setattr(flow, "_last_input_tick", lambda: 10)
    clip = {"v": "previous"}
    title = {"v": "note.py — Visual Studio Code"}

    def paste():
        return clip["v"]

    def copy(text):
        clip["v"] = text

    def send(keys):
        if keys == "ctrl+c":
            clip["v"] = copied["v"]

    copied = {"v": "the line\n"}
    monkeypatch.setattr(flow.pyperclip, "paste", paste)
    monkeypatch.setattr(flow.pyperclip, "copy", copy)
    monkeypatch.setattr(flow.keyboard, "send", send)
    monkeypatch.setattr(flow, "foreground_window", lambda: title["v"])
    text, _saved, problem = flow.capture_selection()
    assert problem is None and text == "the line\n"

    copied["v"] = "the line\n"
    clip["v"] = "previous"
    title["v"] = "note.txt - Notepad"
    text, _saved, problem = flow.capture_selection()
    assert problem is None and text == "the line\n"

    copied["v"] = "one\ntwo\n"
    clip["v"] = "previous"
    text, _saved, problem = flow.capture_selection()
    assert problem is None and text == "one\ntwo\n"


def test_astra_r2_12_short_release_does_not_wait_for_a_gap(monkeypatch):
    """Removed by design: an ordinary short tap is processed at key-up."""
    import flow
    assert not hasattr(flow.Flow, "_poll_latch")
    f = _flow(monkeypatch)
    slept = []
    monkeypatch.setattr(flow.time, "sleep", lambda s: slept.append(s))
    started = time.monotonic()
    f._latch_event("down", 0.0)
    f._latch_event("up", 0.2)
    assert time.monotonic() - started < 0.15
    assert slept == []
    assert f.recording is False
    assert f._job_q.get_nowait()[0] == "dictate"
    f._latch_event("tick", 0.2 + flow._LATCH_GAP_S)
    assert f._job_q.empty()


# --- Astra round-3 ----------------------------------------------------------


def test_astra_r3_1_summary_with_typed_text_copies_and_never_backspaces(monkeypatch):
    """Live-typed text stays. The summary is copied. Nothing is backspaced."""
    import numpy as np
    import flow
    assert not hasattr(flow.Flow, "_erase_typed")
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    sent = []
    monkeypatch.setattr(flow.keyboard, "send", lambda *a, **k: sent.append(a))
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *a, **k: ("hello there friend", {}))
    monkeypatch.setattr(flow, "llm_summarize", lambda *a, **k: "SHORT")
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 11)
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow, "_last_input_tick", lambda: 1000)
    copied = []
    monkeypatch.setattr(flow, "_copy_clipboard",
                        lambda text: copied.append(text) or True)
    pasted = []
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    said = []
    f.overlay = types.SimpleNamespace(set=lambda state: said.append(state))
    streamer = types.SimpleNamespace(
        typed_len=7, typed_chars=7, stop=lambda: None,
        committed_text="hello t", committed_samples=0,
        saved_clip="OLD",
        restore_clipboard=lambda: copied.append("RESTORED"))
    audio = np.ones(32, dtype=np.float32)
    f.process(audio, streamer=streamer, summary=True, take_hwnd=11,
              input_tick=1000)
    assert sent == []
    assert pasted == []
    assert "SHORT" in copied
    assert "RESTORED" not in copied
    assert ("say", "Summary copied — paste to replace") in said
    assert f.last_text == "SHORT"

    said.clear()
    pasted.clear()
    copied.clear()
    streamer.typed_len = 0
    streamer.typed_chars = 0
    streamer.saved_clip = "OLD"
    f.process(audio, streamer=streamer, summary=True, take_hwnd=11,
              input_tick=1000)
    assert sent == []
    assert pasted and pasted[0][0] == "SHORT"
    assert ("say", "Summary copied — paste to replace") not in said


def test_astra_r3_2_rewrite_paste_requires_the_same_control_and_stamp(monkeypatch):
    """A rewrite is copied, never pasted, including when focus or the stamp moves.

    Capture still stamps after Flow's own Ctrl+C. The result stays on the
    clipboard for the user to paste.
    """
    import flow
    f = _flow(monkeypatch)
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    state = {
        "tick": 1000,
        "clip": "previous",
        "selection": "Selected words",
        "hwnd": 11,
        "focus": 5,
    }
    pasted, said = [], []
    f.overlay = types.SimpleNamespace(set=lambda item: said.append(item))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "_restore_clip", lambda *a, **k: None)
    monkeypatch.setattr(flow, "_last_input_tick", lambda: state["tick"])
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: state["hwnd"])
    monkeypatch.setattr(flow, "focused_control_hwnd", lambda: state["focus"])
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: state["clip"])

    def copy(text):
        state["clip"] = text

    sent = []

    def send(keys):
        sent.append(keys)
        if keys == "ctrl+c":
            state["tick"] += 50
            state["clip"] = state["selection"]

    monkeypatch.setattr(flow.pyperclip, "copy", copy)
    monkeypatch.setattr(flow.keyboard, "send", send)

    def rewrite(_instruction, selection, _cfg):
        assert selection == "Selected words"
        return "Brief."

    monkeypatch.setattr(flow, "llm_rewrite", rewrite)
    assert f._rewrite_selection("be brief", take_hwnd=11) == "Brief."
    assert pasted == []
    assert "ctrl+v" not in sent
    assert state["clip"] == "Brief."
    assert ("say", "Rewritten — press Ctrl+V to replace") in said
    assert state["tick"] > 1000

    def fail(move):
        pasted.clear()
        said.clear()
        sent.clear()
        state.update(tick=1000, clip="previous", selection="Selected words",
                     hwnd=11, focus=5)
        monkeypatch.setattr(flow, "llm_rewrite", move)
        assert f._rewrite_selection("be brief", take_hwnd=11) == "Brief."
        assert pasted == []
        assert "ctrl+v" not in sent
        assert ("say", "Rewritten — press Ctrl+V to replace") in said
        assert state["clip"] == "Brief."

    def move_focus(*_a, **_k):
        state["focus"] = 99
        return "Brief."

    def user_types(*_a, **_k):
        state["tick"] += 10
        return "Brief."

    def change_text(*_a, **_k):
        state["selection"] = "Other words"
        return "Brief."

    fail(move_focus)
    fail(user_types)
    fail(change_text)


def test_astra_r3_3_client_fallback_matches_baseline(monkeypatch):
    """A lost client session pastes the full fallback, typed prefix included."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["mode"] = "client"
    f.cfg["min_duration_s"] = 0
    pasted = []
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    prefix = "Hello world."
    fallback = "Hello, world. Please call tomorrow."
    # Baseline arithmetic: finish() is None, so typed resets to 0 and the
    # whole fallback is the remainder.
    typed = len(prefix)
    text = None
    if text is None:
        text, typed = fallback, 0
    assert (text or "")[typed:] == fallback
    streamer = flow.ClientStreamer(f)
    streamer.sid = "s"
    streamer.typed = len(prefix)
    streamer.finish = lambda: (None, streamer.typed)
    streamer.restore_clipboard = lambda: None
    monkeypatch.setattr(f, "client_transcribe", lambda *a, **k: fallback)
    f.process(np.ones(16, dtype=np.float32), streamer=streamer)
    assert f.last_text == fallback
    assert pasted == [(fallback, "clipboard")]
    copied = []
    monkeypatch.setattr(flow, "_copy_clipboard",
                        lambda text: copied.append(text) or True)
    f._copy_last_dictation()
    assert copied == [fallback]


def test_astra_r3_4_alt_first_does_not_start_rewrite(monkeypatch):
    """Alt down, ctrl down, both up before the ctrl thread: still a command.

    Ctrl is not left held, so the next ctrl press is a new recording.
    """
    import flow
    f = _flow(monkeypatch)
    f.recording = False
    f._on_key(_ev("right alt", flow.keyboard.KEY_DOWN))
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    f._on_key(_ev("right alt", flow.keyboard.KEY_UP))
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_UP))
    items = _queued(f)
    assert ("rewrite", None) not in items
    assert items == [("down", "command", "right alt"), ("up", "command", "right alt")]
    assert "right ctrl" not in f._pressed_keys
    assert f._active_ptt_key is None
    _play_ctrl(f)
    assert f.recording is False
    assert f.rewrite_take is False
    kind, _args = f._job_q.get_nowait()
    assert kind == "command"
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.get_nowait() == ("down", "dictate", "right ctrl")
    f.recording = True
    f.active_mode = "dictate"
    f.rewrite_take = False
    f._on_key(_ev("left alt", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.get_nowait() == ("rewrite", None)
    client = _flow(monkeypatch)
    client.cfg["mode"] = "client"
    client.recording = True
    client.active_mode = "dictate"
    client._active_ptt_key = "right ctrl"
    client._pressed_keys.add("right ctrl")
    client._on_key(_ev("right alt", flow.keyboard.KEY_DOWN))
    assert ("rewrite", None) not in _queued(client)


def test_astra_r3_5_client_mode_does_not_apply_scratch(monkeypatch):
    """Scratch-that is not applied on the client. The phrase is pasted whole."""
    import inspect
    import numpy as np
    import flow
    import streaming
    src = inspect.getsource(streaming.StreamSession.add_audio)
    assert "note_published" not in src
    f = _flow(monkeypatch)
    f.cfg["mode"] = "client"
    f.cfg["min_duration_s"] = 0
    pasted = []
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a[0]))
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    said = "Drop me. Scratch that. Keep this."
    monkeypatch.setattr(f, "client_transcribe", lambda *a, **k: said)
    f.process(np.ones(16, dtype=np.float32))
    assert pasted == [said]
    assert f.last_text == said
    streamer = flow.ClientStreamer(f)
    streamer.sid = "s"
    streamer.typed = len("Drop me.")
    streamer.finish = lambda: ("Drop me. Scratch that.", streamer.typed)
    streamer.restore_clipboard = lambda: None
    pasted.clear()
    f.process(np.ones(16, dtype=np.float32), streamer=streamer)
    assert f.last_text == "Drop me. Scratch that."
    assert pasted == [" Scratch that."]


def test_astra_r3_6_pending_chord_does_not_convert_the_next_command(monkeypatch):
    """Rewrite before recording starts does not convert this or the next command."""
    import flow
    f = _flow(monkeypatch)
    assert not hasattr(f, "_chord_pending")
    f._enable_rewrite()
    assert f.rewrite_take is False
    f.on_press("command")
    assert f.recording is True
    assert f.active_mode == "command"
    assert f.rewrite_take is False
    f.on_release()
    assert f.recording is False
    f._on_key(_ev("right alt", flow.keyboard.KEY_DOWN))
    assert f._ctrl_q.get_nowait() == ("down", "command", "right alt")
    f.on_press("command")
    assert f.active_mode == "command"
    assert f.rewrite_take is False


def test_astra_r3_7_clipboard_retention_is_per_job(monkeypatch):
    """A copied rewrite does not suppress clipboard restore on the next take."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *a, **k: ("be brief", {}))
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    monkeypatch.setattr(flow, "focused_control_hwnd", lambda: 5)
    monkeypatch.setattr(flow, "_last_input_tick", lambda: 1000)
    copied = []
    monkeypatch.setattr(flow, "_copy_clipboard",
                        lambda text: copied.append(text) or True)
    hwnd = {"v": 11}
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: hwnd["v"])
    monkeypatch.setattr(flow, "capture_selection",
                        lambda: ("Selected words", "prev", None))

    def move(*_a, **_k):
        hwnd["v"] = 22
        return "BRIEF"

    monkeypatch.setattr(flow, "llm_rewrite", move)
    said = []
    f.overlay = types.SimpleNamespace(set=lambda state: said.append(state))
    f.process(np.ones(16, dtype=np.float32), streamer=None, rewrite=True,
              take_hwnd=11)
    assert ("say", "Rewritten — press Ctrl+V to replace") in said
    assert copied[-1] == "BRIEF"
    assert getattr(f, "_clipboard_result", None) is None

    copied.clear()
    hwnd["v"] = 11
    restored = []
    f._clipboard_result = "STALE"
    streamer = types.SimpleNamespace(
        typed_len=0, typed_chars=0, stop=lambda: None,
        committed_text="", committed_samples=0,
        saved_clip="USER CLIP",
        restore_clipboard=lambda: restored.append("USER CLIP"))
    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *a, **k: ("hello there", {}))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: None)
    f.process(np.ones(16, dtype=np.float32), streamer=streamer, rewrite=False)
    assert restored == ["USER CLIP"]
    assert streamer.saved_clip == "USER CLIP"


def test_astra_r3_8_vscode_whole_line_is_a_selection(monkeypatch):
    """A VS Code line ending in newline is a selection. Copy-line is documented."""
    import flow
    assert not hasattr(flow, "_vscode_copy_line")
    doc = flow.capture_selection.__doc__ or ""
    assert "VS Code" in doc
    assert "known limitation" in doc
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 1)
    monkeypatch.setattr(flow, "focused_control_hwnd", lambda: 2)
    monkeypatch.setattr(flow, "_last_input_tick", lambda: 10)
    monkeypatch.setattr(flow, "foreground_window",
                        lambda: "note.py — Visual Studio Code")
    clip = {"v": "previous"}
    copied = {"v": "selected line\n"}

    def paste():
        return clip["v"]

    def copy(text):
        clip["v"] = text

    def send(keys):
        if keys == "ctrl+c":
            clip["v"] = copied["v"]

    monkeypatch.setattr(flow.pyperclip, "paste", paste)
    monkeypatch.setattr(flow.pyperclip, "copy", copy)
    monkeypatch.setattr(flow.keyboard, "send", send)
    text, _saved, problem = flow.capture_selection()
    assert problem is None
    assert text == "selected line\n"


# --- Astra round-4 ----------------------------------------------------------


def _selection_state(monkeypatch, flow, f, sleep=None):
    """Clipboard, focus and tick doubles for a rewrite capture."""
    state = {
        "tick": 1000,
        "clip": "previous",
        "selection": "Selected words",
        "hwnd": 11,
        "focus": 5,
    }
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    monkeypatch.setattr(flow, "_last_input_tick", lambda: state["tick"])
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: state["hwnd"])
    monkeypatch.setattr(flow, "focused_control_hwnd", lambda: state["focus"])
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: state["clip"])
    monkeypatch.setattr(flow.pyperclip, "copy", lambda text: state.__setitem__("clip", text))

    def send(keys):
        if keys == "ctrl+c":
            state["tick"] += 50
            state["clip"] = state["selection"]

    monkeypatch.setattr(flow.keyboard, "send", send)
    monkeypatch.setattr(flow.time, "sleep", sleep or (lambda _s: None))
    return state


def test_astra_r4_1_asr_selection_change_is_not_the_target(monkeypatch):
    """Select A, release, then select B while ASR runs. Do not paste over B."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    said, pasted = [], []
    f.overlay = types.SimpleNamespace(set=lambda item: said.append(item))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    state = _selection_state(monkeypatch, flow, f)
    seen = {}

    def transcribe(*_a, **_k):
        state["selection"] = "Unrelated text"
        state["tick"] += 10
        return ("make it short", {})

    def rewrite(_instruction, selection, _cfg):
        seen["selection"] = selection
        return "Brief."

    monkeypatch.setattr(f, "transcribe_and_store", transcribe)
    monkeypatch.setattr(flow, "llm_rewrite", rewrite)
    f.process(np.ones(32, dtype=np.float32), rewrite=True, take_hwnd=11)
    assert seen["selection"] == "Selected words"
    assert pasted == []
    assert state["selection"] == "Unrelated text"
    assert state["clip"] == "Brief."
    assert ("say", "Rewritten — press Ctrl+V to replace") in said


def test_astra_r4_1_move_during_verify_wait_does_not_paste(monkeypatch):
    """One capture, then the rewrite stays on the clipboard. Nothing is pasted."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    said, pasted = [], []
    f.overlay = types.SimpleNamespace(set=lambda item: said.append(item))
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    sleeps = {"n": 0}

    def sleep(s):
        if s == 0.08:
            sleeps["n"] += 1
            state["selection"] = "Unrelated text"
            state["focus"] = 77
            state["tick"] += 25

    state = _selection_state(monkeypatch, flow, f, sleep=sleep)
    seen = {}

    def rewrite(_instruction, selection, _cfg):
        seen["selection"] = selection
        return "Brief."

    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *_a, **_k: ("make it short", {}))
    monkeypatch.setattr(flow, "llm_rewrite", rewrite)
    f.process(np.ones(32, dtype=np.float32), rewrite=True, take_hwnd=11)
    assert sleeps["n"] == 1
    assert seen["selection"] == "Selected words"
    assert state["selection"] == "Unrelated text"
    assert state["focus"] == 77
    assert pasted == []
    assert state["clip"] == "Brief."
    assert ("say", "Rewritten — press Ctrl+V to replace") in said


def test_astra_r4_2_remote_session_scratch_stays_baseline(monkeypatch):
    """Drop me. / Scratch that. Keep this. through session publish and client delta.

    The thin-client screen stays the unedited text. A local hands-free
    worker still applies scratch-that.
    """
    import flow
    import streaming
    f = _flow(monkeypatch)
    pasted = []
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(flow.pyperclip, "copy", lambda text: pasted.append(text))
    monkeypatch.setattr(flow.keyboard, "send", lambda *a, **k: None)

    def quiet_start(self):
        self._wake.clear()
        self._active = True

    monkeypatch.setattr(flow.StreamWorker, "start", quiet_start)
    session = streaming.StreamSession(f, "phone")
    assert session.worker.voice_edits is False
    session.worker._commit_phrase("Drop me.", 160, [])
    client = flow.ClientStreamer(f)
    client.committed = session.add_audio(b"\x00\x00")
    client._type_delta()
    session.worker._commit_phrase("Scratch that. Keep this.", 160, [])
    client.committed = session.add_audio(b"\x00\x00")
    client._type_delta()
    screen = "".join(p for p in pasted if isinstance(p, str))
    assert screen == "Drop me. Scratch that. Keep this."
    assert "Drop me.s." not in screen
    assert client.committed[client.typed:] == ""

    local = flow.StreamWorker(f, live=False)
    local._active = True
    assert local.voice_edits is True
    local._commit_phrase("Drop me.", 160, [])
    local._commit_phrase("Scratch that. Keep this.", 160, [])
    assert local.committed_text == "Keep this."


def test_astra_r4_3_stop_does_not_arm_the_next_short_take(monkeypatch):
    """Taps 0.00-0.10 and 0.30-0.40 latch. Stop 0.45-0.46. 0.65-0.75 does not."""
    import flow
    f = _flow(monkeypatch)
    clock = {"t": 0.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])

    def tap(down_at, up_at):
        clock["t"] = down_at
        f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
        _play_ctrl(f)
        clock["t"] = up_at
        f._on_key(_ev("right ctrl", flow.keyboard.KEY_UP))
        _play_ctrl(f)

    tap(0.00, 0.10)
    assert f.recording is False and f.latched is False
    tap(0.30, 0.40)
    assert f.latched is True and f.recording is True
    tap(0.45, 0.46)
    assert f.latched is False and f.recording is False
    assert f._latch.get("prev_up_at") is None
    assert f._latch.get("prev_press_s") is None
    tap(0.65, 0.75)
    assert f.latched is False
    assert f.recording is False
    assert f._latch.get("phase") == "idle"


def test_astra_r4_4_older_restore_does_not_overwrite_a_retained_summary(monkeypatch):
    """A rewrite's 0.6s restore must not replace a later copied summary."""
    import numpy as np
    import threading
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    said = []
    f.overlay = types.SimpleNamespace(set=lambda item: said.append(item))
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(flow.keyboard, "press", lambda *_a, **_k: None)
    monkeypatch.setattr(flow.keyboard, "release", lambda *_a, **_k: None)
    gate = threading.Event()

    def sleep(s):
        if s and s >= 0.5:
            assert gate.wait(2)

    state = _selection_state(monkeypatch, flow, f, sleep=sleep)
    state["clip"] = "ORIGINAL"
    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *_a, **_k: ("be brief", {}))
    monkeypatch.setattr(flow, "llm_rewrite", lambda *_a, **_k: "Brief.")
    f.process(np.ones(32, dtype=np.float32), rewrite=True, take_hwnd=11)
    assert state["clip"] == "Brief."

    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *_a, **_k: ("hello there friend", {}))
    monkeypatch.setattr(flow, "llm_summarize", lambda *_a, **_k: "SUMMARY")
    streamer = types.SimpleNamespace(
        typed_len=5, typed_chars=5, stop=lambda: None,
        committed_text="hello", committed_samples=0,
        saved_clip="ORIGINAL",
        restore_clipboard=lambda: state.__setitem__("clip", "RESTORED"))
    f.process(np.ones(32, dtype=np.float32), streamer=streamer, summary=True,
              take_hwnd=11, input_tick=state["tick"])
    assert state["clip"] == "SUMMARY"
    assert ("say", "Summary copied — paste to replace") in said
    gate.set()
    # Let the delayed restore observe the newer generation and return.
    import time as time_mod
    time_mod.sleep(0.2)
    assert state["clip"] == "SUMMARY"
    assert streamer.saved_clip is None


# --- Astra round-5 ----------------------------------------------------------


def test_astra_r5_2_vm_bridge_scratch_stays_baseline(monkeypatch):
    """Drop me. / Scratch that. Keep this. through remote_start and /record/partial.

    The VM types committed[typed:]. A voice edit that shortens the prefix
    yields Drop me.s. voice_edits is off for this worker. Local hands-free
    still edits.
    """
    import flow
    f = _flow(monkeypatch)
    f.cfg["streaming"]["enabled"] = True
    f._remote_gen = 0
    rec = _recorder(monkeypatch)
    rec._stream = types.SimpleNamespace(active=True)
    f.recorder = rec

    def quiet_start(self):
        self._wake.clear()
        self._active = True

    monkeypatch.setattr(flow.StreamWorker, "start", quiet_start)
    assert f.remote_start() is True
    worker = f.remote_streamer
    try:
        assert worker is not None
        assert worker.voice_edits is False

        typed = 0
        screen = []

        def poll():
            # server.py /record/partial, then vm/daemon.py LiveTyper delta.
            nonlocal typed
            st = getattr(f, "remote_streamer", None)
            committed = st.committed_text if st else ""
            delta = committed[typed:]
            if delta:
                screen.append(delta)
                typed = len(committed)

        worker._commit_phrase("Drop me.", 160, [])
        poll()
        worker._commit_phrase("Scratch that. Keep this.", 160, [])
        poll()
        text = "".join(screen)
        assert text == "Drop me. Scratch that. Keep this."
        assert "Drop me.s." not in text
        assert worker.committed_text[typed:] == ""

        local = flow.StreamWorker(f, live=False)
        local._active = True
        assert local.voice_edits is True
        local._commit_phrase("Drop me.", 160, [])
        local._commit_phrase("Scratch that. Keep this.", 160, [])
        assert local.committed_text == "Keep this."
    finally:
        f.remote_cancel()


def test_astra_r5_3_older_restore_during_live_paste_is_a_noop(monkeypatch):
    """An older restore that fires during the 20 ms paste wait must not win.

    The new take's ctrl+v pastes NEW DICTATION and typed_len advances.
    """
    import flow
    f = _flow(monkeypatch)
    clip = {"v": "ORIGINAL"}
    sent = []
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: clip["v"])
    monkeypatch.setattr(flow.pyperclip, "copy",
                        lambda text: clip.__setitem__("v", text))
    monkeypatch.setattr(flow.keyboard, "send",
                        lambda keys: sent.append((keys, clip["v"])))
    prev = flow.StreamWorker(f, live=True)
    prev.saved_clip = "ORIGINAL"
    stale = {"gen": flow._clipboard_generation()}

    def sleep(s):
        if s == 0.02:
            prev.restore_clipboard(stale["gen"])

    monkeypatch.setattr(flow.time, "sleep", sleep)
    worker = flow.StreamWorker(f, live=True)
    worker._active = True
    worker.committed_text = "NEW DICTATION"
    worker.typed_len = 0
    worker.type_chunk()
    assert sent == [("ctrl+v", "NEW DICTATION")]
    assert clip["v"] == "NEW DICTATION"
    assert worker.typed_len == len("NEW DICTATION")

    # The same generation guard covers the client streamer and copy-last.
    stale["gen"] = flow._clipboard_generation()
    client = flow.ClientStreamer(f)
    client.committed = "MORE WORDS"
    client.typed = 0
    client._type_delta()
    assert clip["v"] == "MORE WORDS"
    assert client.typed == len("MORE WORDS")
    prev.restore_clipboard(stale["gen"])
    assert clip["v"] == "MORE WORDS"

    stale["gen"] = flow._clipboard_generation()
    f.last_text = "LAST"
    f.overlay = None
    f._copy_last_dictation()
    assert clip["v"] == "LAST"
    prev.restore_clipboard(stale["gen"])
    assert clip["v"] == "LAST"

    # A restore armed after the write still runs.
    now = flow._clipboard_generation()
    prev.saved_clip = "ORIGINAL"
    prev.restore_clipboard(now)
    assert clip["v"] == "ORIGINAL"


def test_astra_r5_4_asr_error_restores_the_captured_clipboard(monkeypatch):
    """Streaming off. Capture saves ORIGINAL. ASR raises. Clipboard is ORIGINAL."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    f.cfg["streaming"]["enabled"] = False
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    state = _selection_state(monkeypatch, flow, f)
    state["clip"] = "ORIGINAL"

    def boom(*_a, **_k):
        raise RuntimeError("asr failed")

    monkeypatch.setattr(f, "transcribe_and_store", boom)
    with pytest.raises(RuntimeError):
        f.process(np.ones(32, dtype=np.float32), rewrite=True, streamer=None)
    assert state["clip"] == "ORIGINAL"


def test_astra_r5_5_rejected_command_up_does_not_stop_the_take(monkeypatch):
    """Latch, arm summary, tap right alt. The latched take keeps recording.

    A customized command key tapped during an ordinary latch is ignored too.
    The key that owns the take can still stop it.
    """
    import flow
    f = _flow(monkeypatch)
    clock = {"t": 0.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])

    def tap(flow_obj, name, down_at, up_at):
        clock["t"] = down_at
        flow_obj._on_key(_ev(name, flow.keyboard.KEY_DOWN))
        _play_ctrl(flow_obj)
        clock["t"] = up_at
        flow_obj._on_key(_ev(name, flow.keyboard.KEY_UP))
        _play_ctrl(flow_obj)

    tap(f, "right ctrl", 0.00, 0.10)
    assert f.recording is False and f.latched is False
    while not f._job_q.empty():
        f._job_q.get_nowait()
    tap(f, "right ctrl", 0.30, 0.40)
    assert f.latched is True and f.recording is True
    clock["t"] = 0.50
    f._on_key(_ev("right shift", flow.keyboard.KEY_DOWN))
    _play_ctrl(f)
    assert f.summary_take is True
    tap(f, "right alt", 0.55, 0.60)
    assert f.recording is True
    assert f.latched is True
    assert f.summary_take is True
    assert f._job_q.empty()

    other = _flow(monkeypatch)
    other.cfg["command_hotkey"] = "f8"
    tap(other, "right ctrl", 0.00, 0.10)
    while not other._job_q.empty():
        other._job_q.get_nowait()
    tap(other, "right ctrl", 0.30, 0.40)
    assert other.latched is True and other.recording is True
    tap(other, "f8", 0.50, 0.55)
    assert other.recording is True
    assert other.latched is True
    assert other._job_q.empty()
    tap(other, "right ctrl", 0.70, 0.80)
    assert other.recording is False
    assert other.latched is False


# --- Astra round-6 ----------------------------------------------------------


def test_astra_r6_1_rewrite_failure_does_not_restore_over_the_next_paste(monkeypatch):
    """Older rewrite ASR failure during the next take's 20 ms wait.

    The restore was armed at capture. It must not put ORIGINAL back over
    NEW DICTATION, and ctrl+v pastes the new dictation.
    """
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    state = {"clip": "ORIGINAL", "pasted": []}
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: state["clip"])
    monkeypatch.setattr(flow.pyperclip, "copy",
                        lambda text: state.__setitem__("clip", text))

    def send(keys):
        if keys == "ctrl+c":
            state["clip"] = "Selected words"
        elif keys == "ctrl+v":
            state["pasted"].append(state["clip"])

    monkeypatch.setattr(flow.keyboard, "send", send)
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()

    def asr(*_a, **_k):
        entered.set()
        assert release.wait(3)
        raise RuntimeError("ASR failed")

    monkeypatch.setattr(f, "transcribe_and_store", asr)
    errors = []

    def old_job():
        try:
            f.process(np.ones(32, dtype=np.float32), rewrite=True)
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            done.set()

    def sleep(n):
        if n == 0.02:
            release.set()
            assert done.wait(3)

    monkeypatch.setattr(flow.time, "sleep", sleep)
    old = threading.Thread(target=old_job)
    old.start()
    assert entered.wait(3)
    assert state["clip"] == "Selected words"
    worker = flow.StreamWorker(f, live=True)
    worker._active = True
    worker.committed_text = "NEW DICTATION"
    worker.typed_len = 0
    worker.type_chunk()
    old.join(3)
    assert errors == ["ASR failed"]
    assert state["pasted"] == ["NEW DICTATION"]
    assert state["clip"] == "NEW DICTATION"
    assert worker.typed_len == len("NEW DICTATION")


def test_astra_r6_2_queued_taps_keep_take_ownership(monkeypatch):
    """Release, down, release queued before the ctrl thread drains.

    The latch's owner is the key that was accepted, so a right-alt tap
    does not stop it.
    """
    import flow
    f = _flow(monkeypatch)
    clock = {"t": 0.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])

    def event(name, kind, t, drain=True):
        clock["t"] = t
        f._on_key(_ev(name, kind))
        if drain:
            _play_ctrl(f)

    event("right ctrl", flow.keyboard.KEY_DOWN, 0.0)
    assert f.recording is True
    assert f._take_key == "right ctrl"
    event("right ctrl", flow.keyboard.KEY_UP, 0.10, drain=False)
    event("right ctrl", flow.keyboard.KEY_DOWN, 0.30, drain=False)
    event("right ctrl", flow.keyboard.KEY_UP, 0.40, drain=False)
    _play_ctrl(f)
    assert f.recording is True
    assert f.latched is True
    assert f._take_key == "right ctrl"
    assert f._job_q.qsize() == 1
    event("right shift", flow.keyboard.KEY_DOWN, 0.50)
    event("right alt", flow.keyboard.KEY_DOWN, 0.55)
    event("right alt", flow.keyboard.KEY_UP, 0.60)
    assert f.recording is True
    assert f.latched is True
    assert f.summary_take is True
    assert f._take_key == "right ctrl"
    assert f._job_q.qsize() == 1


def test_astra_r6_3_failed_start_clears_only_that_press(monkeypatch):
    """Mic timeout and an open failure drop that press's owner.

    A later right-alt press and a Hermes press still record. A claim that
    a later press installed during the timeout stays.
    """
    import flow
    f = _flow(monkeypatch)
    clock = {"t": 0.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])
    f._mic_lock = types.SimpleNamespace(
        acquire=lambda timeout=None: False, release=lambda: None)
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    _play_ctrl(f)
    clock["t"] = 1.2
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_UP))
    _play_ctrl(f)
    assert f.recording is False
    assert f._latch["phase"] == "idle"
    assert f._take_key is None
    assert f._active_ptt_key is None

    f._mic_lock = flow._new_mic_lock()
    f.recorder = types.SimpleNamespace(start=lambda: 1, stop=lambda: b"",
                                       level=0.0)
    clock["t"] = 2.0
    f._on_key(_ev("right alt", flow.keyboard.KEY_DOWN))
    _play_ctrl(f)
    assert f.recording is True
    assert f.active_mode == "command"
    assert f._take_key == "right alt"
    clock["t"] = 2.2
    f._on_key(_ev("right alt", flow.keyboard.KEY_UP))
    _play_ctrl(f)
    assert f.recording is False
    assert f._take_key is None

    f.cfg["hermes"] = {"enabled": True, "hotkey": "right shift"}
    f.hermes = types.SimpleNamespace(predict_title=lambda _w: "Title")
    clock["t"] = 3.0
    f._on_key(_ev("right shift", flow.keyboard.KEY_DOWN))
    _play_ctrl(f)
    assert f.recording is True
    assert f.active_mode == "hermes"
    assert f._take_key == "right shift"

    held = _flow(monkeypatch)
    monkeypatch.setattr(flow.time, "monotonic", lambda: 0.0)

    def acquire(timeout=None):
        held._claim_owner("right alt")
        return False

    held._mic_lock = types.SimpleNamespace(acquire=acquire, release=lambda: None)
    held._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    _play_ctrl(held)
    held._on_key(_ev("right ctrl", flow.keyboard.KEY_UP))
    _play_ctrl(held)
    assert held.recording is False
    assert held._take_key == "right alt"
    assert held._take_gen == 2

    broken = _flow(monkeypatch)

    def boom():
        raise OSError("mic did not open")

    broken.recorder = types.SimpleNamespace(
        start=boom, stop=lambda: b"", level=0.0)
    broken._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    with pytest.raises(OSError):
        _play_ctrl(broken)
    broken._on_key(_ev("right ctrl", flow.keyboard.KEY_UP))
    _play_ctrl(broken)
    assert broken.recording is False
    assert broken._take_key is None
    assert broken._active_ptt_key is None
    assert broken._latch["phase"] == "idle"
    broken.recorder = types.SimpleNamespace(start=lambda: 1, stop=lambda: b"",
                                            level=0.0)
    broken.cfg["hermes"] = {"enabled": True, "hotkey": "right shift"}
    broken.hermes = types.SimpleNamespace(predict_title=lambda _w: "Title")
    broken._on_key(_ev("right shift", flow.keyboard.KEY_DOWN))
    _play_ctrl(broken)
    assert broken.recording is True
    assert broken.active_mode == "hermes"


def test_astra_r6_4_queued_rewrite_uses_the_release_selection(monkeypatch):
    """The ctrl thread records the release HWND and does not capture.

    Capture runs on the job worker, with the other clipboard transactions.
    The same window rewrites whatever is selected when that job captures.
    A different foreground window cancels and leaves the clipboard alone.
    """
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    state = _selection_state(monkeypatch, flow, f)
    state["hwnd"] = 11
    state["selection"] = "A at release"
    state["clip"] = "ORIGINAL"
    said = []
    f.overlay = types.SimpleNamespace(set=lambda item: said.append(item))
    calls = {"n": 0}
    waits = {"n": 0}
    phase = {"at": "submit"}
    real = flow.capture_selection

    def wrapped():
        if phase["at"] == "submit":
            raise AssertionError("control thread captured the selection")
        calls["n"] += 1
        return real()

    def wait(*_a, **_k):
        if phase["at"] == "submit":
            raise AssertionError("control thread waited on modifiers")
        waits["n"] += 1
        return True

    monkeypatch.setattr(flow, "capture_selection", wrapped)
    monkeypatch.setattr(flow, "wait_modifiers_up", wait)
    seen = []
    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *_a, **_k: ("be brief", {}))
    monkeypatch.setattr(flow, "llm_rewrite",
                        lambda _instruction, selection, _cfg: seen.append(selection) or "Brief.")
    f._submit_job("dictate", np.ones(32, dtype=np.float32), None, None,
                  rewrite=True)
    assert calls["n"] == 0
    assert state["clip"] == "ORIGINAL"
    kind, args = f._job_q.get_nowait()
    assert kind == "dictate"
    assert args[-2] is None
    assert args[-1] == 11
    phase["at"] = "job"
    state["selection"] = "B after release"
    f.process(*args)
    assert calls["n"] == 1
    assert seen == ["B after release"]
    assert state["clip"] == "Brief."

    said.clear()
    phase["at"] = "submit"
    state["hwnd"] = 11
    state["clip"] = "ORIGINAL"
    state["selection"] = "A at release"
    before = calls["n"]
    f._submit_job("dictate", np.ones(32, dtype=np.float32), None, None,
                  rewrite=True)
    assert calls["n"] == before
    assert state["clip"] == "ORIGINAL"
    kind, args = f._job_q.get_nowait()
    assert args[-1] == 11
    phase["at"] = "job"
    state["hwnd"] = 99
    state["selection"] = "B after release"
    waited = waits["n"]
    f.process(*args)
    assert calls["n"] == before
    assert waits["n"] == waited
    assert state["clip"] == "ORIGINAL"
    assert seen == ["B after release"]
    assert ("say", "Selection changed — rewrite cancelled") in said


def test_astra_r7_3_queued_owner_release_accepts_the_next_key(monkeypatch):
    """Alt up, then ctrl down/up, queued before the ctrl thread drains.

    The observed release of an ordinary owner lets the next key through,
    so both the command job and the dictation hold are queued. A latched
    take still rejects a secondary key.
    """
    import flow
    f = _flow(monkeypatch)
    clock = {"t": 0.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])

    def event(name, kind, t, drain=True):
        clock["t"] = t
        f._on_key(_ev(name, kind))
        if drain:
            _play_ctrl(f)

    event("right alt", flow.keyboard.KEY_DOWN, 0.0)
    assert f.recording is True
    assert f.active_mode == "command"
    assert f._take_key == "right alt"
    assert f.latched is False
    event("right alt", flow.keyboard.KEY_UP, 0.10, drain=False)
    event("right ctrl", flow.keyboard.KEY_DOWN, 0.30, drain=False)
    event("right ctrl", flow.keyboard.KEY_UP, 0.40, drain=False)
    queued = _queued(f)
    assert queued == [
        ("up", "command", "right alt"),
        ("down", "dictate", "right ctrl"),
        ("up", "dictate", "right ctrl"),
    ]
    _play_ctrl(f)
    kinds = []
    while True:
        try:
            kinds.append(f._job_q.get_nowait()[0])
        except queue.Empty:
            break
    assert kinds == ["command", "dictate"]
    assert f.recording is False
    assert f._take_key is None

    event("right ctrl", flow.keyboard.KEY_DOWN, 2.00)
    event("right ctrl", flow.keyboard.KEY_UP, 2.10)
    event("right ctrl", flow.keyboard.KEY_DOWN, 2.30)
    event("right ctrl", flow.keyboard.KEY_UP, 2.40)
    assert f.recording is True
    assert f.latched is True
    assert f._take_key == "right ctrl"
    assert f._job_q.qsize() == 1
    event("right shift", flow.keyboard.KEY_DOWN, 2.50)
    assert f.summary_take is True
    event("right alt", flow.keyboard.KEY_DOWN, 2.55, drain=False)
    event("right alt", flow.keyboard.KEY_UP, 2.60, drain=False)
    assert _queued(f) == []
    assert f.recording is True
    assert f.latched is True
    assert f.summary_take is True
    assert f._take_key == "right ctrl"
    assert f._job_q.qsize() == 1


def test_astra_r7_4_streamed_tail_restores_the_original_clipboard(monkeypatch):
    """Live 'Hello' plus the final ' world' puts the user's clipboard back."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    f.cfg["paste_mode"] = "clipboard"
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    monkeypatch.setattr(flow.keyboard, "press", lambda *_a, **_k: None)
    monkeypatch.setattr(flow.keyboard, "release", lambda *_a, **_k: None)
    monkeypatch.setattr(flow.keyboard, "send", lambda *_a, **_k: None)
    state = {"clip": "ORIGINAL"}
    copied = []

    def copy(text):
        state["clip"] = text
        copied.append(text)

    monkeypatch.setattr(flow.pyperclip, "paste", lambda: state["clip"])
    monkeypatch.setattr(flow.pyperclip, "copy", copy)
    worker = flow.StreamWorker(f, live=True)
    assert worker.saved_clip == "ORIGINAL"
    worker._active = True
    worker.committed_text = "Hello"
    worker.typed_len = 0
    worker.type_chunk()
    assert state["clip"] == "Hello"
    assert worker.typed_len == len("Hello")
    monkeypatch.setattr(
        f, "transcribe_and_store", lambda *_a, **_k: ("Hello world", {}))
    f.process(np.ones(32, dtype=np.float32), streamer=worker,
              committed_at_release=("Hello", 0))
    assert " world" in copied
    assert state["clip"] == "ORIGINAL"


def test_astra_r7_5_early_rewrite_returns_restore_the_capture(monkeypatch):
    """A supplied rewrite capture is restored on a short take and a deaf mic."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    state = {"clip": "ORIGINAL"}
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: state["clip"])
    monkeypatch.setattr(
        flow.pyperclip, "copy", lambda text: state.__setitem__("clip", text))
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    captures = {"n": 0}

    def capture():
        captures["n"] += 1
        raise AssertionError("early return must not capture")

    monkeypatch.setattr(flow, "capture_selection", capture)

    def arm():
        gen = flow._copy_clipboard("SELECTED")
        assert type(gen) is int
        assert state["clip"] == "SELECTED"
        return ({"text": "SELECTED", "saved": "ORIGINAL", "gen": gen}, None)

    f.cfg["min_duration_s"] = 5
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    f.process(np.ones(16, dtype=np.float32), rewrite=True,
              rewrite_capture=arm())
    assert state["clip"] == "ORIGINAL"
    assert captures["n"] == 0

    f.cfg["min_duration_s"] = 0
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: True)
    f.process(np.ones(1600, dtype=np.float32), rewrite=True,
              rewrite_capture=arm())
    assert state["clip"] == "ORIGINAL"
    assert captures["n"] == 0

    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(
        f, "transcribe_and_store", lambda *_a, **_k: ("", {}))
    f.process(np.ones(1600, dtype=np.float32), rewrite=True,
              rewrite_capture=arm())
    assert state["clip"] == "ORIGINAL"
    assert captures["n"] == 0


def test_astra_r8_1_disabled_alt_never_captures_a_selection(monkeypatch):
    """Rewrite off: alt during a take does not capture or touch the clipboard.

    Alt pressed first stays a command. The hint line drops Alt rewrite.
    A rewrite=True job is also a no-op while the flag is off.
    """
    import numpy as np
    import flow
    assert flow.DEFAULTS["rewrite"]["enabled"] is False
    assert "Alt rewrite" not in flow._HINT_LINE
    f = _flow(monkeypatch)
    f.cfg["rewrite"] = {"enabled": False}
    f.cfg["min_duration_s"] = 0
    clock = {"t": 20.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])
    said = []
    f.overlay = types.SimpleNamespace(set=lambda item: said.append(item))
    calls = {"n": 0}
    writes = []

    def capture():
        calls["n"] += 1
        raise AssertionError("capture_selection while rewrite is disabled")

    def copy(*args, **kwargs):
        writes.append(args)
        raise AssertionError("clipboard write while rewrite is disabled")

    monkeypatch.setattr(flow, "capture_selection", capture)
    monkeypatch.setattr(flow, "_copy_clipboard", copy)
    monkeypatch.setattr(flow.pyperclip, "copy", lambda text: writes.append(text))
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: "USER CLIPBOARD")
    monkeypatch.setattr(flow, "paste_text", lambda *a, **k: None)
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 7)
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(f, "transcribe_and_store", lambda *a, **k: ("hello", {}))

    def at(t, name, kind):
        clock["t"] = t
        f._on_key(_ev(name, kind))
        _play_ctrl(f)

    at(20.0, "right alt", flow.keyboard.KEY_DOWN)
    assert f.recording is True
    assert f.active_mode == "command"
    assert f.rewrite_take is False
    at(20.5, "right alt", flow.keyboard.KEY_UP)
    assert f.recording is False
    assert calls["n"] == 0

    at(22.0, "right ctrl", flow.keyboard.KEY_DOWN)
    assert f.recording is True
    assert f.active_mode == "dictate"
    assert f.rewrite_take is False
    before = list(said)
    for name in ("left alt", "right alt", "alt"):
        clock["t"] += 0.05
        f._on_key(_ev(name, flow.keyboard.KEY_DOWN))
        assert _queued(f) == []
        assert f.rewrite_take is False
        assert said == before
    f._enable_rewrite()
    assert f.rewrite_take is False
    assert said == before
    assert calls["n"] == 0
    assert writes == []
    at(22.6, "right ctrl", flow.keyboard.KEY_UP)
    assert f.recording is False
    jobs = []
    while True:
        try:
            jobs.append(f._job_q.get_nowait())
        except queue.Empty:
            break
    assert [job[0] for job in jobs] == ["command", "dictate"]
    assert jobs[1][1][4] is False
    f.process(*jobs[1][1])
    assert calls["n"] == 0
    assert writes == []
    f.process(np.ones(1600, dtype=np.float32), rewrite=True)
    assert calls["n"] == 0
    assert writes == []
    assert f.rewrite_take is False


def test_astra_r8_2_ctrl_down_between_handle_up_and_release_is_accepted(monkeypatch):
    """Right ctrl down after _handle_up starts, before on_release, is queued.

    The release marker stays until the alt claim is cleared, and both change
    under the same lock. A lock release must not show the marker gone while
    alt still owns the take.
    """
    import flow
    f = _flow(monkeypatch)
    clock = {"t": 0.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])

    class _Watch:
        def __init__(self):
            self._inner = threading.Lock()
            self.splits = []
            self._before = (None, None)

        def __enter__(self):
            self._inner.acquire()
            self._before = (
                getattr(f, "_take_key", None),
                getattr(f, "_owner_up_queued", None),
            )
            return self

        def __exit__(self, *exc):
            take = getattr(f, "_take_key", None)
            marker = getattr(f, "_owner_up_queued", None)
            _before_take, before_marker = self._before
            if before_marker == "right alt" and marker is None and take == "right alt":
                self.splits.append((before_marker, take, marker))
            self._inner.release()
            return False

    watch = _Watch()
    f._key_lock = watch

    def event(name, kind, t, drain=True):
        clock["t"] = t
        f._on_key(_ev(name, kind))
        if drain:
            _play_ctrl(f)

    event("right alt", flow.keyboard.KEY_DOWN, 0.0)
    assert f.recording is True
    assert f._take_key == "right alt"
    event("right alt", flow.keyboard.KEY_UP, 0.10, drain=False)
    assert f._owner_up_queued == "right alt"
    assert f._take_key == "right alt"
    seen = {}
    real = f.on_release

    def release(key=None):
        assert f._owner_up_queued == "right alt"
        assert f._take_key == "right alt"
        seen["marker_at_down"] = f._owner_up_queued
        seen["owner_at_down"] = f._take_key
        clock["t"] = 0.20
        f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
        seen["queued"] = _queued(f)
        f.on_release = real
        real(key)

    f.on_release = release
    _play_ctrl(f)
    assert ("down", "dictate", "right ctrl") in seen["queued"]
    assert seen["marker_at_down"] == "right alt"
    assert seen["owner_at_down"] == "right alt"
    assert watch.splits == []
    assert f.recording is True
    assert f._take_key == "right ctrl"
    assert f._owner_up_queued is None
    event("right ctrl", flow.keyboard.KEY_UP, 0.80)
    assert f.recording is False
    assert f._take_key is None
    assert f._owner_up_queued is None
    assert watch.splits == []
    kinds = []
    while True:
        try:
            kinds.append(f._job_q.get_nowait()[0])
        except queue.Empty:
            break
    assert kinds == ["command", "dictate"]


def test_astra_r8_5_scratch_that_updates_stored_cleaned(monkeypatch):
    """Scratch-that edits the stored cleaned text. raw stays the decoder text."""
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    f.cfg["cleanup"] = {"enabled": False}
    f.cfg["trusted_devices"] = ["android-bubble"]
    f.cfg["model"] = "fake"
    f.cfg["device"] = "cpu"
    f.cfg["vocab"] = []
    f.cfg["replacements"] = {}
    heard = "Drop me. Scratch that. Keep this."
    rows = []

    def decode(_audio, beam_size=None, prompt=None, word_timestamps=False):
        return [types.SimpleNamespace(text=heard)]

    f.decode = decode

    class _Store:
        def add(self, raw, **kwargs):
            rows.append((raw, dict(kwargs)))
            return {"id": "row"}

    f.store = _Store()
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    pasted = []
    monkeypatch.setattr(
        flow, "paste_text",
        lambda text, mode, restore=True: pasted.append(text))
    f.process(np.ones(1600, dtype=np.float32))
    assert len(rows) == 1
    raw, kwargs = rows[0]
    assert raw == heard
    assert kwargs["cleaned"] == "Keep this."
    assert pasted == ["Keep this."]


# --- Astra round-9 ----------------------------------------------------------


def test_astra_r9_1_watchdog_release_queues_the_next_key(monkeypatch):
    """Right alt down and up during watchdog release of right ctrl are queued.

    The owner-release marker is published in the same lock hold as the drop
    and stays until that claim clears. The record cap does the same. A
    latched watchdog disarm does not publish the marker and does not stop.
    """
    import flow
    f = _flow(monkeypatch)
    f.cfg["rewrite"] = {"enabled": False}
    clock = {"t": 0.0}
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(flow, "_os_key_down", lambda vk: False)

    class _Watch:
        def __init__(self):
            self._inner = threading.Lock()
            self.splits = []
            self._before_active = None

        def __enter__(self):
            self._inner.acquire()
            self._before_active = getattr(f, "_active_ptt_key", None)
            return self

        def __exit__(self, *exc):
            active = getattr(f, "_active_ptt_key", None)
            marker = getattr(f, "_owner_up_queued", None)
            take = getattr(f, "_take_key", None)
            if (self._before_active == "right ctrl" and active is None
                    and marker is None and take == "right ctrl"
                    and not getattr(f, "latched", False)):
                self.splits.append((self._before_active, active, marker, take))
            self._inner.release()
            return False

    watch = _Watch()
    f._key_lock = watch

    def event(name, kind, t, drain=True):
        clock["t"] = t
        f._on_key(_ev(name, kind))
        if drain:
            _play_ctrl(f)

    def release_during(key=None):
        assert f._owner_up_queued == "right ctrl"
        assert f._take_key == "right ctrl"
        assert f._active_ptt_key is None
        clock["t"] = clock["t"] + 0.2
        f._on_key(_ev("right alt", flow.keyboard.KEY_DOWN))
        clock["t"] = clock["t"] + 0.1
        f._on_key(_ev("right alt", flow.keyboard.KEY_UP))
        seen["queued"] = _queued(f)
        assert f._owner_up_queued == "right ctrl"
        assert f._take_key == "right ctrl"
        f.on_release = real
        real(key)

    seen = {}
    real = f.on_release
    event("right ctrl", flow.keyboard.KEY_DOWN, 0.0)
    assert f.recording is True
    assert f._take_key == "right ctrl"
    assert f._active_ptt_key == "right ctrl"
    f.on_release = release_during
    f._poll_key_watchdog()
    assert f._active_ptt_key == "right ctrl"
    assert seen == {}
    f._poll_key_watchdog()
    assert seen["queued"] == [
        ("down", "command", "right alt"),
        ("up", "command", "right alt"),
    ]
    assert f.recording is False
    assert f._take_key is None
    assert f._owner_up_queued is None
    assert watch.splits == []

    while True:
        try:
            f._ctrl_q.get_nowait()
        except queue.Empty:
            break
    f._ptt_times.clear()

    seen.clear()
    event("right ctrl", flow.keyboard.KEY_DOWN, 5.0)
    assert f.recording is True
    assert f._take_key == "right ctrl"
    f._record_started_at = 0.0
    clock["t"] = 700.0
    f.on_release = release_during
    f._poll_record_cap()
    assert seen["queued"] == [
        ("down", "command", "right alt"),
        ("up", "command", "right alt"),
    ]
    assert f.recording is False
    assert f._take_key is None
    assert f._owner_up_queued is None
    assert watch.splits == []

    while True:
        try:
            f._ctrl_q.get_nowait()
        except queue.Empty:
            break
    f._ptt_times.clear()
    event("right ctrl", flow.keyboard.KEY_DOWN, 10.0)
    event("right ctrl", flow.keyboard.KEY_UP, 10.1)
    event("right ctrl", flow.keyboard.KEY_DOWN, 10.3)
    event("right ctrl", flow.keyboard.KEY_UP, 10.4)
    assert f.latched is True
    assert f.recording is True
    f._pressed_keys.add("right ctrl")
    f._active_ptt_key = "right ctrl"
    f._owner_up_queued = None
    f._os_up_streak = 0

    def refuse(*_a, **_k):
        raise AssertionError("latched watchdog disarm must not release")

    f.on_release = refuse
    f._poll_key_watchdog()
    f._poll_key_watchdog()
    assert f.recording is True
    assert f.latched is True
    assert f._active_ptt_key is None
    assert "right ctrl" not in f._pressed_keys
    assert f._owner_up_queued is None
    assert f._take_key == "right ctrl"
    assert watch.splits == []


def test_astra_r9_2_failed_summary_copy_keeps_the_text(monkeypatch):
    """Ten clipboard failures leave the summary in memory.

    The overlay says to use Copy last dictation. It does not say the
    summary was copied. The clipboard stays UNRELATED CLIPBOARD.
    """
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(f, "transcribe_and_store",
                        lambda *a, **k: ("hello there friend", {}))
    summary = {"text": "SHORT"}
    monkeypatch.setattr(flow, "llm_summarize", lambda *a, **k: summary["text"])
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    hwnd = {"v": 11}
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: hwnd["v"])
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow, "_last_input_tick", lambda: 1000)
    monkeypatch.setattr(flow.keyboard, "send", lambda *a, **k: None)
    state = {"clip": "UNRELATED CLIPBOARD"}
    attempts = {"n": 0}

    def copy(_text):
        attempts["n"] += 1
        raise OSError("clipboard busy")

    monkeypatch.setattr(flow.pyperclip, "copy", copy)
    monkeypatch.setattr(flow.pyperclip, "paste", lambda: state["clip"])
    said = []
    f.overlay = types.SimpleNamespace(set=lambda item: said.append(item))
    pasted = []
    monkeypatch.setattr(f, "_safe_paste", lambda *a, **k: pasted.append(a))
    streamer = types.SimpleNamespace(
        typed_len=7, typed_chars=7, stop=lambda: None,
        committed_text="hello t", committed_samples=0,
        saved_clip="UNRELATED CLIPBOARD", clip_gen=None,
        restore_clipboard=lambda *a, **k: None)
    audio = np.ones(32, dtype=np.float32)
    f.process(audio, streamer=streamer, summary=True, take_hwnd=11,
              input_tick=1000)
    assert attempts["n"] == 10
    assert state["clip"] == "UNRELATED CLIPBOARD"
    assert flow.pyperclip.paste() == "UNRELATED CLIPBOARD"
    assert f.last_text == "SHORT"
    assert ("say", "Summary ready — use Copy last dictation") in said
    assert ("say", "Summary copied — paste to replace") not in said
    assert pasted == []

    said.clear()
    pasted.clear()
    summary["text"] = "SHORTER"
    hwnd["v"] = 99
    streamer.typed_len = 0
    streamer.typed_chars = 0
    before = attempts["n"]
    f.process(audio, streamer=streamer, summary=True, take_hwnd=11,
              input_tick=1000)
    assert attempts["n"] == before + 10
    assert state["clip"] == "UNRELATED CLIPBOARD"
    assert f.last_text == "SHORTER"
    assert ("say", "Summary ready — use Copy last dictation") in said
    assert ("say", "Summary copied — window changed") not in said
    assert ("say", "Summary copied — paste to replace") not in said
    assert pasted == []


def test_astra_r9_3_streaming_scratch_keeps_the_raw_transcript(monkeypatch):
    """Ghost/hands-free scratch edits cleaned text and leaves raw intact.

    Commit `Drop me.` then `Scratch that. Keep this.` Raw history keeps
    the pre-edit transcript frozen at release. Cleaned text is `Keep this.`
    """
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    f.cfg["min_duration_s"] = 0
    f.cfg["cleanup"] = {"enabled": False}
    f.cfg["trusted_devices"] = ["android-bubble"]
    f.cfg["model"] = "fake"
    f.cfg["device"] = "cpu"
    f.cfg["vocab"] = []
    f.cfg["replacements"] = {}
    f.cfg["normalize_gain"] = False
    monkeypatch.setattr(f, "mic_is_deaf", lambda _a: False)
    monkeypatch.setattr(flow, "foreground_window", lambda: "Editor")
    monkeypatch.setattr(flow.time, "sleep", lambda _s: None)
    pasted = []
    monkeypatch.setattr(
        flow, "paste_text",
        lambda text, mode, restore=True: pasted.append(text))

    def decode(*_a, **_k):
        raise AssertionError("committed audio is not a tail")

    f.decode = decode
    rows = []

    class _Store:
        def add(self, raw, **kwargs):
            rows.append((raw, dict(kwargs)))
            return {"id": "row"}

    f.store = _Store()
    rec = _recorder(monkeypatch)
    rec._recording = True
    rec._chunks = [np.ones((1600, 1), dtype=np.float32)]
    f.recorder = rec
    worker = flow.StreamWorker(f, live=False)
    worker._active = True
    worker.source = types.SimpleNamespace(_buf_lock=None)
    worker._commit_phrase("Drop me.", 800, [])
    worker._commit_phrase("Scratch that. Keep this.", 800, [])
    assert worker.committed_text == "Keep this."
    assert worker.committed_raw == "Drop me. Scratch that. Keep this."
    audio, committed = f._capture_take(worker)
    assert worker._frozen == ("Keep this.", 1600)
    assert worker._frozen_raw == "Drop me. Scratch that. Keep this."
    assert committed == (
        "Keep this.", 1600, "Drop me. Scratch that. Keep this.")
    worker._commit_phrase("Nope.", 10, [])
    assert worker.committed_raw == "Drop me. Scratch that. Keep this."
    assert worker._frozen_raw == "Drop me. Scratch that. Keep this."
    f.process(audio, streamer=worker, committed_at_release=committed,
              full_samples=rec.last_full_samples,
              discarded=rec.last_discarded)
    assert len(rows) == 1
    raw, kwargs = rows[0]
    assert raw == "Drop me. Scratch that. Keep this."
    assert kwargs["cleaned"] == "Keep this."
    assert pasted == ["Keep this."]
