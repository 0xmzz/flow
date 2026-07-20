# Live mode ("Jarvis") — v1 spec

Goal: a third input mode for Flow. Toggle a hotkey once and Flow listens
continuously; each spoken utterance that starts with the wake word is routed
through the existing command router (`commands.py Router.handle`) and fires
**fast** — target ≤300ms from end-of-speech to action (p50), ≤500ms p95.
Inspired by CNVS's Parakeet setup; entirely local on beast (RTX 5090).

## Non-goals (v1)

- No server routes, no thin-client/phone/VM exposure (design LiveWorker so a
  server route COULD drive it later, but build none of it).
- No agent orchestration verbs (comes later).
- No cloud anything. `local_only` semantics untouched.
- Whisper dictation/command paths (f8 / right alt) unchanged.

## ASR engine

- NVIDIA **Parakeet-TDT-0.6b-v3** via the `onnx-asr` pip package
  (`pip install onnx-asr[gpu,hub]` into the existing `.venv`, Python 3.12).
  Model auto-downloads from HF on first use; cache it under `data/models/`
  (gitignored) via the package's cache dir mechanism, NOT the repo root.
- Execution provider: try CUDA (`onnxruntime-gpu`); if session creation or a
  warmup decode fails (Blackwell/sm_120 wheels can be finicky), fall back to
  CPU int8 automatically and log which provider is live. Do a warmup decode at
  startup so the first real utterance isn't slow.
- Measure and log per-utterance decode ms + realtime factor. If CPU fallback
  can't hit the latency budget, that's acceptable for v1 — log it loudly,
  don't block the build on it.
- Whisper large-v3 stays loaded and untouched; Parakeet is a second, separate
  model. VRAM headroom on the 5090 is ample. Parakeet decodes do NOT need
  `model_lock` (that lock serializes the whisper GPU model only), but keep
  decodes serialized within LiveWorker itself.

## VAD + segmentation

- Streaming Silero VAD, ONNX (no torch dependency — if `onnx-asr` bundles a
  silero VAD wrapper, use it; otherwise `pysilero-vad` or raw
  onnxruntime with the silero v5 onnx). CPU, 16kHz, 32ms frames.
- Utterance = speech run ending with `live.silence_ms` (default **280**) of
  non-speech, padded with ~150ms pre/post. Hard cap `live.max_utterance_s`
  (default 12) — force-close and decode.
- Feed: subscribe to the existing always-open `Recorder` stream (extend it
  with a lightweight subscriber/queue on its `_on_audio` callback). Do NOT
  open a second input stream.

## Activation + precedence

- New config key `live_hotkey`, default **f9**, registered in `_on_key`
  alongside the existing keys. Press = toggle live mode on/off (not hold).
- While a PTT session is active (f8 or right alt held) or a remote session
  owns the recorder, LiveWorker pauses (drops audio, no segmentation) and
  resumes after. PTT always wins.
- Live mode state survives nothing: app restart = off.

## Wake word gate

- `live.wake_word` default `"flow"`. An utterance is acted on only if its
  first token fuzzy-matches the wake word (difflib ratio ≥0.75 against the
  first word, lowercased, punctuation stripped — "flow", "flo", "floh" pass).
  Wake word is stripped before routing: "flow open youtube" → `handle("open
  youtube")`.
- Non-matching utterances: discard audio immediately, log one line
  (`[live] ignored: <text>`), do NOT store to history, do NOT speak.
- `live.wake_word: null` disables the gate (open mic — for headphone use).

## Routing + feedback

- Matched utterances go to `self.router.handle(text)` exactly like
  `process_command` does today; store to history with `source="pc-live"`
  (respect existing store semantics), speak the response via SAPI, and show it
  on the overlay pill.
- Overlay: add a `live` idle-listening state (distinct color from recording)
  and reuse the existing "say" flash for responses. Tray icon: new state or
  tooltip suffix showing live mode is on. Keep it minimal — no new UI beyond
  states.
- Startup print + tray tooltip mention the toggle key.

## Config defaults (add to DEFAULTS in flow.py)

```python
"live_hotkey": "f9",
"live": {
    "enabled": True,          # master switch for the feature
    "wake_word": "flow",
    "silence_ms": 280,
    "max_utterance_s": 12,
    "engine": "nemo-parakeet-tdt-0.6b-v3",
},
```

## Files

- New module `live.py` — LiveWorker (VAD loop, segmentation, Parakeet decode,
  wake gate) + ParakeetEngine (load/warmup/decode/provider fallback).
  Keyboard-free: flow.py owns the hotkey and calls
  `live_worker.start()/stop()/pause()/resume()`.
- `flow.py` — hotkey wiring, Recorder subscriber hook, overlay/tray states,
  config defaults. Keep the diff surgical.
- `experiments/live_latency.py` — offline harness: feed WAV files (e.g.
  `last.wav` or `data/dataset` samples) through VAD→Parakeet, print per-stage
  ms and end-of-speech→text latency. This is the acceptance evidence.
- `tests/test_invariants.py` — add: (1) wake-word gate accepts/strips/rejects
  correctly incl. fuzzy variants; (2) ignored utterances never reach
  store.add; (3) LiveWorker never constructs any network client (privacy
  invariant — decode path is pure local).

## Hard edges (handle explicitly)

- **Self-hearing:** while SAPI speaks a response (and while whisper PTT
  feedback plays), the mic hears it. Pause segmentation (drop frames + reset
  VAD state) for the duration of `speak()` calls triggered by live mode, and
  resume with a clean buffer.
- **PTT precedence must be atomic:** f8/right-alt press while an utterance is
  mid-capture → discard the partial utterance, reset VAD, pause. On PTT
  release, resume with drained buffers (no stale audio replay) and restore the
  live overlay/tray state.
- **f9 and PTT key state are independent** — toggling live mode while a PTT
  key is held must not wedge either state machine.

## Acceptance

1. `python -m pytest tests -q` green.
2. `python experiments/live_latency.py` runs and prints decode ms per sample +
   which provider (CUDA/CPU) is active.
3. `python flow.py` starts with both whisper and parakeet warm, f9 toggles
   live mode, f8/right-alt behavior unchanged. (Human live-mic test happens
   after merge — don't attempt to use the real microphone yourself.)
4. No new network egress in the speech path (model download at install time
   excepted).
