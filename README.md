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

## Autostart

Create a shortcut to `.venv\Scripts\pythonw.exe flow.py` in
`shell:startup`.

## Roadmap

- Android port (whisper.cpp / futo-style keyboard integration)
- Streaming partials, per-app tone profiles
