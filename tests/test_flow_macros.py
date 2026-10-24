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
    assert act == "seal" and st["phase"] == "pending"
    st, act = flow.latch_step(st, "down", 10.5)        # 0.3s after the release
    assert act == "second" and st["phase"] == "second"
    st, act = flow.latch_step(st, "up", 10.6)          # short second release latches
    assert act == "latch" and st["phase"] == "latched"
    st, act = flow.latch_step(st, "down", 20.0)        # single tap while latched
    assert act == "stop" and st["phase"] == "idle"


def test_short_tap_without_a_second_press_releases():
    """A <0.35s tap that nobody follows still ends the take."""
    import flow
    st = _idle()
    st, _ = flow.latch_step(st, "down", 1.0)
    st, act = flow.latch_step(st, "up", 1.2)
    assert act == "seal" and st["phase"] == "pending"
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
    text, _typed = streamer.finish(sealed)
    assert text == "said"
    audio_posts = [p for p in posts[before:] if "/audio" in p[0]]
    assert len(audio_posts) == 1
    assert len(audio_posts[0][1]) == 3000 * 2
    assert trimmed == []

    seen = {}

    def fake_finish(audio=None):
        seen["finish_audio"] = audio
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
    assert seen["finish_audio"] is sealed
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
    assert ("say", "Rewrite copied — window changed") in said

    erased = []
    monkeypatch.setattr(f, "_erase_typed", lambda n: erased.append(n))
    streamer = types.SimpleNamespace(typed_chars=12)
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: False)
    assert f._erase_summary_prefix(streamer, 12, 11) == (0, True)
    assert erased == []
    assert ("say", "Summary pasted — original left in place") in said
    said.clear()
    monkeypatch.setattr(flow, "wait_modifiers_up", lambda *a, **k: True)
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 22)
    assert f._erase_summary_prefix(streamer, 12, 11) == (0, True)
    assert erased == []
    assert ("say", "Summary pasted — original left in place") in said
    monkeypatch.setattr(flow, "foreground_hwnd", lambda: 11)
    assert f._erase_summary_prefix(streamer, 12, 11) == (0, False)
    assert erased == [12]


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
    streamer = flow.ClientStreamer(f)
    streamer.committed = "replacement"
    streamer.typed = 0
    f.rewrite_take = True
    streamer._type_delta()
    f.rewrite_take = False
    f.latched = True
    streamer._type_delta()
    assert sent == []
    assert streamer.typed == 0
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
    import numpy as np
    import flow
    st = _idle()
    st, _ = flow.latch_step(st, "down", 0.0)
    st, act = flow.latch_step(st, "up", 0.10)
    assert act == "seal" and st["phase"] == "pending"
    st, act = flow.latch_step(st, "down", 0.10 + flow._LATCH_GAP_S)
    assert act == "restart"  # a gap of exactly 0.40s is not a latch
    st = _idle()
    st, _ = flow.latch_step(st, "down", 0.0)
    st, _ = flow.latch_step(st, "up", 0.10)
    st, act = flow.latch_step(st, "down", 0.30)
    assert act == "second" and st["phase"] == "second"
    assert act != "latch"
    # Same float as the threshold, measured from down_at 0, so the compare is exact.
    border = {"phase": "second", "down_at": 0.0, "up_at": None}
    held_out, act = flow.latch_step(border, "up", flow._LATCH_MAX_PRESS_S)
    assert act == "release" and held_out["phase"] == "idle"
    short = {"phase": "second", "down_at": 0.0, "up_at": None}
    latched, act = flow.latch_step(short, "up", 0.20)
    assert act == "latch" and latched["phase"] == "latched"

    f = _flow(monkeypatch)
    rec = _recorder(monkeypatch)
    rec._stream = types.SimpleNamespace(active=True)
    f.recorder = rec
    f._latch_event("down", 0.0)
    rec._on_audio(np.ones((15, 1), dtype=np.float32))
    f._latch_event("up", 0.10)
    f._latch_event("down", 0.20)
    assert f._latch["phase"] == "second"
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


def test_astra_8_alt_first_rewrite_ends_on_ctrl_release(monkeypatch):
    import flow
    f = _flow(monkeypatch, recording=True)
    f.active_mode = "command"
    f._alt_held = True
    f._active_ptt_key = "right alt"
    f._pressed_keys.add("right alt")
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_DOWN))
    assert f._active_ptt_key == "right ctrl"
    assert "right ctrl" in f._pressed_keys
    assert f._plain_release is True
    assert f._ctrl_q.get_nowait() == ("rewrite", None)
    f._on_key(_ev("right alt", flow.keyboard.KEY_UP))
    assert f._ctrl_q.empty()
    f._on_key(_ev("right ctrl", flow.keyboard.KEY_UP))
    assert f._ctrl_q.get_nowait() == ("up", "dictate")
    released = []
    monkeypatch.setattr(f, "on_release", lambda: released.append("up"))
    f._handle_up("dictate", 3.0)
    assert released == ["up"]
    assert f._plain_release is False
    assert f._latch["phase"] != "latched"


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
    f._sealed = {"audio": b"x"}
    assert f.remote_cancel() is True
    assert f.latched is False
    assert f._latch["phase"] == "idle"
    assert f._sealed is None
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
    assert f._ctrl_q.get_nowait() == ("down", "dictate")
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
                        lambda saved, delay=0.0: restored.append(saved))
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
    import numpy as np
    import flow
    f = _flow(monkeypatch)
    rec = _recorder(monkeypatch)
    rec._stream = types.SimpleNamespace(active=True)
    f.recorder = rec
    f._latch_event("down", 0.0)
    rec._on_audio(np.ones((40, 1), dtype=np.float32))
    f._latch_event("up", 0.2)
    assert f.recording is True
    assert f._live_buffer is False
    sealed = np.array(f._sealed["audio"])
    assert len(sealed) == 40
    rec._on_audio(np.full((25, 1), 9, dtype=np.float32))
    f._latch_event("tick", 0.2 + flow._LATCH_GAP_S)
    assert f.recording is False
    kind, args = f._job_q.get_nowait()
    assert kind == "dictate"
    assert np.array_equal(np.asarray(args[0]), sealed)
    assert 9 not in np.asarray(args[0])
    assert f._job_q.empty()

    f2 = _flow(monkeypatch)
    rec2 = _recorder(monkeypatch)
    rec2._stream = types.SimpleNamespace(active=True)
    f2.recorder = rec2
    f2._latch_event("down", 0.0)
    rec2._on_audio(np.full((40, 1), 1, dtype=np.float32))
    f2._latch_event("up", 0.1)
    sealed2 = np.array(f2._sealed["audio"])
    rec2._on_audio(np.full((10, 1), 9, dtype=np.float32))
    f2._latch_event("down", 0.2)
    rec2._on_audio(np.full((10, 1), 8, dtype=np.float32))
    f2._latch_event("up", 0.3)
    assert f2.latched is True
    assert f2.recording is True
    snap = rec2.snapshot()
    assert np.array_equal(snap, sealed2)
    assert 8 not in snap and 9 not in snap
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
