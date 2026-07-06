# Flow — architecture

Self-hosted voice layer: push-to-talk dictation and voice commands on Windows,
phone dictation over Tailscale, all inference on the local GPU. Nothing leaves
your hardware except two explicit, documented exceptions (see [Privacy](#privacy-boundaries)).

## System map

```mermaid
flowchart LR
  subgraph beast[beast — Windows PC, RTX 5090]
    subgraph app[flow.py]
      HK[keyboard hook<br/>right ctrl = dictate<br/>right shift = command] --> REC[Recorder<br/>16k mono f32]
      REC --> SW[StreamWorker<br/>commit-while-held]
      SW --> ASR[faster-whisper large-v3<br/>fp16 CUDA]
      ASR --> CL[llm.py chain<br/>qwen3:8b via Ollama<br/>→ z.ai fallback]
      CL --> PASTE[clipboard paste<br/>at cursor]
      CL --> CMD[commands.py Router<br/>price / open / search / LLM]
      CMD --> TTS[SAPI speak + overlay]
      OV[Overlay pill<br/>listening / transcribing / answer]
      TRAY[pystray menu]
    end
    ASR --> ST[(store.py<br/>data/history.jsonl<br/>data/dataset/*.wav)]
    SRV[server.py :8756<br/>token-gated HTTP] --> ASR
    SRV --> ST
  end
  PHONE[Android: bubble app v0.x<br/>+ whisper-to-input IME] -->|WAV/m4a over Tailscale| SRV
  MV[multivac ~/flow-context/sweep.py<br/>hourly cron :15] -->|/api/history?since=cursor| SRV
  MV -->|/llm — summarize on beast GPU| SRV
  MV --> HB[(Hermes MEMORY.md pointer<br/>+ imports/flow-context-DAY.md<br/>+ vault/memory canon)]
```

## Dictation pipeline (the latency story)

```mermaid
sequenceDiagram
  participant U as You (hold right ctrl)
  participant R as Recorder
  participant S as StreamWorker
  participant W as Whisper (5090)
  participant L as qwen3:8b (5090)
  participant A as Active app

  U->>R: hold — audio accumulates
  loop every 1.2s while held
    S->>W: decode uncommitted tail (beam 1, word timestamps)
    Note over S: words identical in 2 consecutive passes<br/>AND ending at a ≥0.35s pause → COMMITTED
  end
  U->>R: release
  S->>W: final decode of remaining tail only (beam 2)
  Note over S: text = committed + tail → ~0.8s even for 30s speech
  alt window matches a style profile (WhatsApp / mail) and ≥5 words
    S->>L: whole-utterance cleanup (~0.5s, think off)
  end
  S->>A: paste at cursor (clipboard saved & restored)
```

Key decisions baked into this path:

- **Commit only at pauses.** Cutting audio mid-phrase mangles formatting
  ("10%" → "10 percent"); committing at ≥0.35s gaps keeps phrases intact.
- **Word-level agreement, not segment-level.** Whisper grows one segment for a
  normal sentence, so segments never stabilize; words do.
- **VAD threshold 0.3 + `large-v3` (not turbo).** The 0.5 default silently
  deleted whispered speech before decode; turbo's 4-layer decoder is weakest
  exactly on quiet/ambiguous audio. (Research-verified, then A/B'd on real
  whisper samples.)
- **Cleanup is whole-utterance, post-release, profile-gated.** Filler-stripping
  needs the full sentence; it runs only for windows where tone matters
  (chat/email profiles), and never for <5-word snippets.

## Command mode

Right shift routes the transcript to `commands.py` instead of pasting:
regex skills first (crypto price via CoinGecko, `open <site/app>`, `search`),
otherwise the local LLM answers. Responses are spoken (SAPI) **and** shown in
the overlay pill. `POST /command` exposes the same router to remote devices.

## LLM chain (`llm.py`)

```
local: Ollama qwen3:8b  (native /api/chat, think:false, keep_alive -1, warmed at startup)
  └─ on failure → fallback block in config (z.ai glm-4.7, anthropic-shape)
       └─ on failure → raw transcript passes through unchanged
```

The Ollama call uses the **native** API because the OpenAI-compat endpoint
drops `think:false` and Qwen3 then burns ~7s on hidden reasoning.

## Server API (`server.py`, port 8756)

| Route | Auth | Purpose |
|---|---|---|
| `GET /` | page public, data via API | dashboard (stats, activity, pipeline, recent) |
| `GET /health` | none | liveness `{ok, model}` |
| `GET /apk`, `GET /apk/version` | none (tailnet-only) | Android self-update channel |
| `POST /transcribe` | Bearer | raw audio body → `{text, asr_s, duration_s}` |
| `POST /v1/audio/transcriptions` | Bearer | OpenAI-Whisper-compatible (multipart, m4a/ogg/wav via PyAV); strict `{text}` response |
| `POST /command` | Bearer | audio → transcript → command router → `{text, response}` |
| `POST /llm` | Bearer | `{system,text}` → local model only (never fallback) — used by the multivac sweep |
| `GET /api/history?limit&since` | Bearer | entries + cursor; `since` marks a delta sync |
| `GET /api/stats`, `GET /api/meta` | Bearer | totals; model/storage/uptime/last-pull |

A `threading.Lock` serializes the GPU between hotkey and server calls.

## Data model

`data/history.jsonl` — one JSON object per dictation:
`{id, ts, text, cleaned, source, window, duration_s, asr_s, words, model, device, audio}`

- **`text` is always the raw transcript** — fillers and hesitations are signal
  about the speaker, kept deliberately. `cleaned` is what was pasted (null if same).
- `data/dataset/<date>/<id>.wav` — paired audio (archive_audio on): the full
  signal (tone, pace) and a future fine-tuning corpus of your own voice.
- Everything under `data/` is gitignored, as are `config.json` (holds the
  server token + fallback API key) and logs.

## Hermes integration (context sweep)

Hourly cron on multivac: pull only new entries (`since=<cursor>`), summarize
**on beast's GPU** via `/llm`, then write:

1. a constant-size 2-line pointer into Hermes's always-injected `MEMORY.md`
   (sentinel-delimited, replaced each run — context cost never grows; pull, not push),
2. the full day brief to `~/.hermes/imports/flow-context-<day>.md` (30-day retention),
3. a permanent copy to `~/vault/memory/`.

If beast is offline the sweep skips; the brief goes stale rather than falling
back to a cloud API.

## Android app (`android/`)

Zero-dependency Kotlin (one androidx artifact for FileProvider). Components:

- **BubbleService** — overlay bubble: hold=record (AudioRecord → WAV →
  `POST /transcribe`), drag=move, blue/red/amber states; hidden unless a
  keyboard is visible (accessibility as the IME detector).
- **FlowAccessibilityService** — inserts text at the cursor, **paste-first**
  (immune to hint-text-as-content, honors cursor position), SET_TEXT merge as
  fallback. Must be `exported="true"` or Android won't list it; sideloads need
  the one-time "Allow restricted settings" unlock.
- **Updater** — checks `/apk/version` on open, downloads and offers install.
- whisper-to-input (third-party IME) works against `/v1/audio/transcriptions`
  as an alternative keyboard-based path.

## Privacy boundaries

Leaves your hardware, ever:

1. **z.ai fallback** — text (never audio) of one dictation, only while Ollama
   is down. Delete `cleanup.fallback` in config for hard-local-only.
2. **CoinGecko / opened sites** — the extracted query of a voice command
   ("bitcoin"), not your speech.

Everything else — audio, transcripts, briefs, the corpus — stays on beast and
multivac. Phone traffic rides Tailscale (WireGuard, E2E). Note: once Hermes
reads the daily brief, Hermes's own cloud model provider sees it — that's a
property of Hermes, not of Flow.

## Ops

- Autostart: `flow.vbs` in the user Startup folder; single-instance lock on
  `127.0.0.1:18756`; startup unmute-recovery (a crash mid-recording used to
  leave system speakers muted).
- Windows firewall: inbound TCP 8756 allowed ("Flow ASR server" rule).
- Latency budget (measured): hold→release ~constant regardless of length;
  release→pasted ≈ 0.7–0.9s raw path, +~0.5s styled windows.
