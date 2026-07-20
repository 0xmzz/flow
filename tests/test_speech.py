import threading

import speech


def reset(monkeypatch):
    monkeypatch.setattr(speech, "_engine", None)
    monkeypatch.setattr(speech, "_provider", "unavailable")
    monkeypatch.setattr(speech, "_warm_state", "cold")
    monkeypatch.setattr(speech, "_cancel_generation", 0)
    monkeypatch.setattr(speech, "_play", None)
    monkeypatch.setattr(speech, "_wait", None)
    monkeypatch.setattr(speech, "_stop_playback", None)


def test_kokoro_failure_uses_sapi(monkeypatch):
    reset(monkeypatch)
    spoken = []
    monkeypatch.setattr(speech, "_sapi", spoken.append)
    monkeypatch.setattr(speech, "_make_engine",
                        lambda: (_ for _ in ()).throw(ImportError("no kokoro")))
    speech.configure({"engine": "kokoro"}, start=False)
    speech._warm()
    speech.speak("fallback")
    assert spoken == ["fallback"]
    assert speech._warm_state == "failed"


def test_engine_config_respected_and_unknown_uses_sapi(monkeypatch):
    reset(monkeypatch)
    spoken = []
    monkeypatch.setattr(speech, "_sapi", spoken.append)
    speech.configure({"engine": "sapi"}, start=False)
    speech.speak("forced")
    speech.configure({"engine": "mystery"}, start=False)
    speech.speak("unknown")
    assert spoken == ["forced", "unknown"]
    assert speech._cfg["engine"] == "sapi"


def test_speaks_are_serialized(monkeypatch):
    reset(monkeypatch)
    first_playing = threading.Event()
    release_first = threading.Event()
    order = []

    class Engine:
        def create(self, text, **_kwargs):
            order.append("synth:" + text)
            return [1.0], 24000

    def play(samples, **_kwargs):
        order.append("play:" + str(float(samples[0])))
        first_playing.set()

    waits = 0
    def wait():
        nonlocal waits
        waits += 1
        if waits == 1:
            release_first.wait(2)
        order.append("done")

    speech.configure({"engine": "kokoro"}, start=False)
    monkeypatch.setattr(speech, "_engine", Engine())
    monkeypatch.setattr(speech, "_warm_state", "ready")
    monkeypatch.setattr(speech, "_play", play)
    monkeypatch.setattr(speech, "_wait", wait)
    t1 = threading.Thread(target=speech.speak, args=("one",))
    t2 = threading.Thread(target=speech.speak, args=("two",))
    t1.start()
    assert first_playing.wait(1)
    t2.start()
    assert order == ["synth:one", "play:1.0"]
    release_first.set()
    t1.join(2)
    t2.join(2)
    assert order == ["synth:one", "play:1.0", "done",
                     "synth:two", "play:1.0", "done"]


def test_stop_aborts_fake_playback(monkeypatch):
    reset(monkeypatch)
    waiting = threading.Event()
    stopped = threading.Event()

    class Engine:
        def create(self, _text, **_kwargs):
            return [0.0], 24000

    def wait():
        waiting.set()
        stopped.wait(2)

    speech.configure({"engine": "kokoro"}, start=False)
    monkeypatch.setattr(speech, "_engine", Engine())
    monkeypatch.setattr(speech, "_warm_state", "ready")
    monkeypatch.setattr(speech, "_play", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(speech, "_wait", wait)
    monkeypatch.setattr(speech, "_stop_playback", stopped.set)
    thread = threading.Thread(target=speech.speak, args=("long",))
    thread.start()
    assert waiting.wait(1)
    speech.stop()
    thread.join(2)
    assert stopped.is_set()
    assert not thread.is_alive()
