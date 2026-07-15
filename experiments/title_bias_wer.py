"""Measure whether biasing faster-whisper with the foreground window title
helps or hurts — the empirical gate for "screen-aware dictation" v1.

No human ground truth exists, so we measure two things that ARE decidable:

  1. HALLUCINATION (the real risk, per the Fable review): clips dictated into
     app windows ("Claude", "ChatGPT", "WhatsApp"...) almost never SAY the app
     name. If injecting the title makes the app name appear in the transcript
     where the baseline didn't have it, that's the title leaking into output —
     a false positive we can count exactly.

  2. CHANGE + PLAUSIBLE-GAIN: how often the title changes the output at all, and
     among changes, cases where a title proper-noun replaces a near-miss token
     in the baseline (a plausible correction) vs pure injection.

Runs identically on the 5090 or the multivac 4080. Reads a manifest.jsonl of
{id, window, wav, text} and a wavs/ dir; writes report.json + prints a verdict.
"""

import json
import os
import re
import sys
from pathlib import Path

# Expose the pip-installed NVIDIA cuBLAS/cuDNN DLLs to ctranslate2 (same as flow.py).
_nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if _nv.is_dir():
    for _b in _nv.glob("*/bin"):
        os.add_dll_directory(str(_b))
        os.environ["PATH"] = str(_b) + os.pathsep + os.environ["PATH"]

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
MANIFEST = BASE / "manifest.jsonl"
WAVS = Path(os.environ["WAVS"]) if os.environ.get("WAVS") else BASE / "wavs"

# Match production ASR settings (flow.py) exactly.
MODEL = os.environ.get("MODEL", "large-v3")
VOCAB = ["Hyperliquid", "HYPE", "Aerodrome", "Velodrome", "AERO", "veNFT",
         "Folionomics", "Parallax", "Refinery", "Derive", "Deribit", "Ramses",
         "multivac", "Hermes", "Wispr Flow", "Dannis", "Dromos", "Linear",
         "Notion", "TWAP", "perps", "stablecoins", "WHOOP", "Zcash", "Codex",
         "Claude Code", "GLM", "Qwen", "Ollama", "Tailscale", "CoinGecko",
         "Whisper", "APK", "Kelly", "FDV", "DRI"]
VAD = {"threshold": 0.3, "min_silence_duration_ms": 400, "speech_pad_ms": 600}

# "Chrome" tokens: app/UI words that appear in window titles but are essentially
# never spoken. Their appearance in a title-biased transcript = injection.
CHROME = ["claude", "chatgpt", "whatsapp", "telegram", "discord", "slack",
          "google", "chrome", "cursor", "gmail", "outlook", "notion", "grok",
          "comet", "new tab", "visual studio", "code", "youtube", "spotify"]


def vocab_prefix():
    return ", ".join(VOCAB) + "."


def sanitize_title(w):
    """v1 candidate: strip app-chrome suffix, URLs, counters — keep content."""
    w = re.sub(r"\s*[—\-|]\s*[^—\-|]*$", "", w or "")   # drop trailing "— App"
    w = re.sub(r"https?://\S+|\(\d+\)|\[\d+\]|\d+ (new|unread)", "", w)
    return w.strip()


def words(s):
    return re.findall(r"[a-z0-9']+", (s or "").lower())


def transcribe(model, wav, prompt):
    segs, _ = model.transcribe(
        str(wav), language="en", vad_filter=True, vad_parameters=VAD,
        beam_size=3, condition_on_previous_text=False,
        no_speech_threshold=None, initial_prompt=prompt or None)
    return " ".join(s.text.strip() for s in segs).strip()


def main():
    from faster_whisper import WhisperModel
    dev = os.environ.get("DEVICE", "cuda")
    print(f"loading {MODEL} on {dev}...", flush=True)
    model = WhisperModel(MODEL, device=dev, compute_type="float16")

    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8") if l.strip()]
    out = []
    inj_raw = inj_san = changed_raw = changed_san = n = 0
    chrome_exposed = 0  # clips whose title contains a chrome token (probe set)

    for i, r in enumerate(rows):
        wav = WAVS / r["wav"]
        if not wav.exists():
            continue
        title = r.get("window", "") or ""
        base = transcribe(model, wav, vocab_prefix())
        raw = transcribe(model, wav, vocab_prefix() + " " + title)
        san = transcribe(model, wav, vocab_prefix() + " " + sanitize_title(title))
        n += 1
        bw = set(words(base))
        title_chrome = [c for c in CHROME if c in title.lower()]
        probe = bool(title_chrome)
        chrome_exposed += probe
        # injection = a chrome token from the title that appears in the biased
        # output but NOT the baseline (i.e., not acoustically there).
        def injected(hyp):
            hw = set(words(hyp))
            return any(c.split()[0] in hw and c.split()[0] not in bw
                       for c in title_chrome)
        ir, isn = injected(raw), injected(san)
        inj_raw += ir and probe
        inj_san += isn and probe
        changed_raw += raw != base
        changed_san += san != base
        rec = {"id": r.get("id"), "window": title[:50],
               "base": base, "raw_title": raw, "san_title": san,
               "probe": probe, "inj_raw": ir, "inj_san": isn}
        out.append(rec)
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    report = {
        "model": MODEL, "clips": n, "chrome_probe_clips": chrome_exposed,
        "change_rate_raw": round(changed_raw / n, 3) if n else 0,
        "change_rate_sanitized": round(changed_san / n, 3) if n else 0,
        "injection_rate_raw": round(inj_raw / chrome_exposed, 3) if chrome_exposed else 0,
        "injection_rate_sanitized": round(inj_san / chrome_exposed, 3) if chrome_exposed else 0,
        "rows": out,
    }
    (BASE / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== VERDICT DATA ===")
    print(f"clips: {n}  (chrome-probe clips: {chrome_exposed})")
    print(f"title changed output:   raw {report['change_rate_raw']:.0%}  "
          f"sanitized {report['change_rate_sanitized']:.0%}")
    print(f"INJECTION (app-name hallucinated into transcript):")
    print(f"  raw title:       {report['injection_rate_raw']:.0%} of probe clips")
    print(f"  sanitized title: {report['injection_rate_sanitized']:.0%} of probe clips")
    print("\nSample injections (raw):")
    for r in out:
        if r["inj_raw"]:
            print(f"  [{r['window']}] base={r['base'][:45]!r} -> {r['raw_title'][:45]!r}")


if __name__ == "__main__":
    main()
