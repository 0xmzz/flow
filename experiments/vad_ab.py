"""Tier-2 VAD threshold A/B on real corpus audio (runs on the multivac 4080).

Question (open since the whisper-tuning session): does lowering the Silero VAD
threshold below 0.3 recover quiet speech we currently drop, without letting
ambient noise hallucinate text?

Metrics per threshold, on the QUIETEST corpus clips + a normal-loudness control:
  - empty rate (dictation lost entirely)  <- what a lower threshold should fix
  - changed-vs-baseline rate
  - mean output chars (inflation = hallucination proxy, esp. on controls)
"""

import json
import os
import sys
import time
from pathlib import Path

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
THRESHOLDS = [0.3, 0.25, 0.2]

from faster_whisper import WhisperModel  # noqa: E402

print("loading large-v3...", flush=True)
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

rows = [json.loads(l) for l in (BASE / "manifest.jsonl").read_text().splitlines() if l.strip()]
out = {th: [] for th in THRESHOLDS}
t0 = time.time()
for i, r in enumerate(rows):
    wav = BASE / "wavs" / r["wav"]
    for th in THRESHOLDS:
        segs, _ = model.transcribe(
            str(wav), language="en", vad_filter=True,
            vad_parameters={"threshold": th, "min_silence_duration_ms": 400,
                            "speech_pad_ms": 600},
            beam_size=3, condition_on_previous_text=False,
            no_speech_threshold=None)
        text = " ".join(s.text.strip() for s in segs).strip()
        out[th].append({"id": r["id"], "grp": r["grp"], "text": text})
    if i % 10 == 0:
        print(f"{i}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

def stats(th, grp):
    xs = [x for x in out[th] if x["grp"] == grp]
    base = {x["id"]: x["text"] for x in out[0.3] if x["grp"] == grp}
    empty = sum(1 for x in xs if not x["text"])
    changed = sum(1 for x in xs if x["text"] != base.get(x["id"], ""))
    mean_len = sum(len(x["text"]) for x in xs) / max(1, len(xs))
    return {"n": len(xs), "empty": empty, "changed_vs_030": changed,
            "mean_chars": round(mean_len, 1)}

report = {str(th): {"quiet": stats(th, "quiet"), "control": stats(th, "control")}
          for th in THRESHOLDS}
(BASE / "vad_report.json").write_text(json.dumps(
    {"summary": report, "outputs": {str(k): v for k, v in out.items()}}, indent=2))
print(json.dumps(report, indent=2))
