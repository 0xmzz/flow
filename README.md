# Flow

Local push-to-talk dictation for Windows. Hold a hotkey, speak, release — the
transcription is pasted at your cursor. A self-hosted Wispr Flow: ASR runs
entirely on your GPU, nothing leaves the machine unless you enable the optional
LLM cleanup pass.

## Stack

- **ASR**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (`large-v3-turbo` by default) on CUDA
- **Hotkey/paste**: `keyboard` global hook + clipboard paste (clipboard restored after)
- **Cleanup (optional)**: any Anthropic or OpenAI-compatible endpoint strips
  filler words and fixes punctuation without rephrasing
- Tray icon: grey = idle, red = recording, amber = transcribing

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install faster-whisper sounddevice numpy keyboard pyperclip pystray pillow requests nvidia-cublas-cu12 nvidia-cudnn-cu12
.venv\Scripts\python flow.py
```

First run downloads the model (~1.6 GB) and writes `config.json`.

## Usage

Hold **F8**, speak, release. Text appears at your cursor.

## Config (`config.json`)

| key | meaning |
|---|---|
| `hotkey` | hold-to-talk key (any `keyboard` lib name) |
| `model` | faster-whisper model id (`large-v3-turbo`, `distil-large-v3`, ...) |
| `language` | `null` = auto-detect, or e.g. `"en"` |
| `cleanup.enabled` | LLM cleanup pass on/off |
| `cleanup.provider` | `anthropic` or `openai` (any OpenAI-compatible `base_url`) |
| `cleanup.api_key_env` | env var holding the API key (default `FLOW_API_KEY`) |
| `paste_mode` | `clipboard` (fast) or `type` (simulated keystrokes) |

## History / data collection

Every dictation is appended to `data/history.jsonl` (id, timestamp, text, source,
duration, model). Nothing leaves the machine.

```powershell
.venv\Scripts\python flow.py --history 30   # last 30 dictations
.venv\Scripts\python flow.py --stats        # totals
```

Set `history.archive_audio: true` to also keep the paired audio under
`data/dataset/<date>/<id>.wav` — a fine-tuning-ready corpus of your own voice.

## Server mode (dictate from your phone)

With `server.enabled: true`, Flow exposes an ASR endpoint on the local GPU.
On first run it prints an auto-generated token (also saved to `config.json`).
Reach it from any device on your Tailscale network:

| route | purpose |
|---|---|
| `POST /transcribe` | wav body → `{"text": ...}` |
| `POST /v1/audio/transcriptions` | OpenAI-Whisper-compatible (multipart `file`) — point any app that supports a custom Whisper endpoint here |
| `GET /api/history` | recent dictations (JSON) |
| `GET /` | browsable history page |
| `GET /health` | liveness (no auth) |

All routes except `/health` and `/` require `Authorization: Bearer <token>`.

```bash
# from the phone, over Tailscale (PC hostname "beast"):
curl -X POST http://beast:8756/v1/audio/transcriptions \
  -H "Authorization: Bearer <token>" -F file=@clip.wav
```

Windows Firewall: allow the port once (elevated PowerShell):

```powershell
New-NetFirewallRule -DisplayName "Flow ASR server" -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 8756 -Profile Any
```

## Autostart

Create a shortcut to `.venv\Scripts\pythonw.exe flow.py` in `shell:startup`.

## Roadmap

- Android client (approach chosen via design pass — see below)
- Streaming partials, per-app tone profiles
