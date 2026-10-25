# Flow

**Voice control for Windows that runs entirely on your machine.**

Hold a key to dictate anywhere. Press F9 to talk to your computer — open things,
arrange windows, ask questions, hand hard jobs to a coding agent. Teach it a
routine once by speaking it, and it replays forever.

No account. No subscription. No audio leaves your PC.

```
you:  "flow, save this layout as start the day"
flow: "Saved 4 windows: chrome monitor 1 right, code monitor 2 full.
       Say 'start the day' any time."

(next morning)
you:  "hey flow"
flow: "Morning. Want me to run start the day?"
you:  "yes"
```

## What it does

**Dictation.** Hold a hotkey, speak, release — text appears at your cursor in
any app. Words commit *while you're still talking*, so a 30-second dictation
lands ~0.8s after you let go. An optional local LLM pass adapts tone per app
(casual in WhatsApp, precise in a terminal).

**Live mode (F9).** Always-listening command mode. Say the wake word once and
you're in a conversation — no repeating it every sentence. End-of-speech to
action is ~0.4s.

- *"open chrome"*, *"close the tab"*, *"navigate to github"*, *"next desktop"*
- *"put chrome on the left half of monitor 2"*
- *"what's the weather"*, *"price of bitcoin"*
- *"type dear sir, thank you for your patience"*

**Workflows it learns from you.** Two ways to teach it:

- *Snapshot* — arrange your screen, then say *"save this layout as work mode"*.
  It reads which windows are open and where each one sits, and writes a recipe
  that rebuilds it.
- *Dialog* — ask for something it doesn't know and it offers: *"I don't know how
  to do that yet. Teach me the steps?"* Speak them one at a time. It saves the
  result under the phrase you originally used.

Workflows can call other workflows, branch on what's open, and end by handing a
real task to a coding agent.

**Agent hand-off.** *"flow, ask codex to add retry logic to the upload script"* —
it confirms, dispatches to a CLI agent on your machine, streams progress into
the panel, and speaks a summary when done. Works with whichever agent CLI you
already use.

## Privacy is the architecture, not a policy

Speech recognition, the assistant brain, and the voice all run on your hardware.
`local_only` is **enforced in code**, not promised in a README: the LLM chain
drops any endpoint that isn't provably loopback before calling it, and tests
fail the build if that stops being true.

The only things that ever leave the machine are the ones you'd expect: a city
name if you ask for weather, a coin name if you ask for a price, and whatever
you explicitly hand to an agent.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 | — |
| Dictation | CPU (slower) | NVIDIA GPU, 6GB+ |
| Live mode | CPU (~0.6s) | NVIDIA GPU (~0.1s) |
| Assistant brain | *optional* — skills and workflows work without it | [Ollama](https://ollama.com) + `qwen3:8b` |

Everything degrades rather than breaks: no GPU falls back to CPU, no Ollama
means the fast paths and workflows still work.

## Install

One step, on Windows with an NVIDIA GPU (see `INSTALL.txt`):

```powershell
git clone https://github.com/0xmzz/flow; cd flow
powershell -ExecutionPolicy Bypass -File install.ps1   # add -Cleanup for local LLM tidy-up
```

It installs Python 3.12, the CUDA wheels and the speech model into
`%LOCALAPPDATA%\Flow`, adds a desktop shortcut and autostart, and launches.

Manual install:

```powershell
git clone https://github.com/0xmzz/flow; cd flow
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python flow.py --setup
```

`--setup` checks your GPU, microphones, and Ollama, then writes a config that
matches your machine. First run downloads the speech models (~2GB).

Then run `.venv\Scripts\pythonw flow.py` — or put `flow.vbs` in your Startup
folder to have it there every boot.

## Under the hood

| Layer | What runs |
|---|---|
| Dictation ASR | faster-whisper `large-v3` (CUDA fp16), streaming commit-at-pauses |
| Live-mode ASR | NVIDIA Parakeet-TDT-0.6b (ONNX), Silero VAD endpointing |
| Brain | Ollama `qwen3:8b` — fast paths first, model only when needed |
| Voice | Kokoro neural TTS (ONNX), SAPI fallback |
| Window control | Win32 via ctypes — no automation dependencies |

The design rule throughout: **known things instantly, understood things in half
a second, hard things handed off.** Every command you use pushes more of your
traffic into the instant tier.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit and why each
decision was made the way it was.

## Beyond the desktop

Flow can also run as a small server on your own network: phone dictation (an
Android bubble app, or any Whisper-compatible keyboard), a Linux-VM bridge, and
thin-client PCs that auto-update from the host. All optional, all off by
default — see ARCHITECTURE.md.

## Command reference

```powershell
.venv\Scripts\python flow.py --setup      # check this machine, write a config
.venv\Scripts\python flow.py --devices    # list microphones with live levels
.venv\Scripts\python flow.py --history 30 # last 30 dictations
.venv\Scripts\python flow.py --stats      # totals
```

Everything else lives in `config.json`: hotkeys, wake word, voice, conversation
window, per-app tone profiles, personal vocabulary, and word replacements.

## Status

Used daily by its author on Windows 11. Interfaces may still move.
Issues and PRs welcome.
