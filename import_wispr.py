"""One-time importer: recover Wispr Flow data into Flow.

Reads Wispr's leftover SQLite (Electron userData) and:
  1. merges its custom dictionary into config.json (vocab / replacements / snippets),
  2. exports dictation history to data/imported-wispr/history.jsonl,
  3. exports paired audio WAVs to data/imported-wispr/audio/ (the fine-tuning corpus).

Idempotent-ish: re-running overwrites the imported-wispr folder and re-merges
the dictionary (dedup). Original Wispr data is only READ.

Usage:  python import_wispr.py [--no-audio]
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).parent
CONFIG_PATH = APP_DIR / "config.json"
WISPR_DIR = Path(os.environ["APPDATA"]) / "Wispr Flow"
OUT_DIR = APP_DIR / "data" / "imported-wispr"
AUDIO_DIR = OUT_DIR / "audio"


def merge_dictionary(cur):
    vocab_add, replacements, snippets = [], {}, {}
    for r in cur.execute(
        "select phrase, replacement, isSnippet from Dictionary "
        "where (isDeleted=0 or isDeleted is null)"
    ):
        phrase, repl, is_snip = r["phrase"], r["replacement"], r["isSnippet"]
        if not phrase:
            continue
        if is_snip and repl and repl != "None":
            snippets[phrase] = repl
        elif repl and repl != "None":
            replacements[phrase] = repl                      # btw -> by the way
        elif "@" not in phrase and len(phrase) <= 40:
            vocab_add.append(phrase)                         # proper nouns -> bias vocab

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    vocab = cfg.get("vocab") or []
    have = {v.lower() for v in vocab}
    for v in vocab_add:
        if v.lower() not in have:
            vocab.append(v)
            have.add(v.lower())
    cfg["vocab"] = vocab
    cfg.setdefault("replacements", {}).update(replacements)
    cfg.setdefault("snippets", {}).update(snippets)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return vocab_add, replacements, snippets


def best_text(row):
    for k in ("editedText", "formattedText", "asrText"):
        if row[k]:
            return row[k]
    return ""


def iso(ts):
    # Wispr stores "2026-07-06 09:01:53.359 +00:00"
    try:
        return datetime.fromisoformat(ts).astimezone(timezone.utc)\
            .isoformat(timespec="seconds")
    except Exception:
        return ts


def main():
    export_audio = "--no-audio" not in sys.argv
    db = WISPR_DIR / "flow.sqlite"
    if not db.exists():
        sys.exit(f"Wispr db not found at {db}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if export_audio:
        AUDIO_DIR.mkdir(exist_ok=True)

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    v, r, s = merge_dictionary(cur)
    print(f"dictionary merged: +{len(v)} vocab {v}, "
          f"{len(r)} replacements {r}, {len(s)} snippets")

    n = n_audio = n_corr = 0
    with (OUT_DIR / "history.jsonl").open("w", encoding="utf-8") as out:
        for row in cur.execute(
            "select transcriptEntityId, timestamp, app, duration, numWords, "
            "asrText, formattedText, editedText, detectedLanguage, audio "
            "from History order by timestamp"
        ):
            rid = row["transcriptEntityId"]
            text = best_text(row)
            if not text:
                continue
            edited = row["editedText"] if (row["editedText"]
                                           and row["editedText"] != row["asrText"]) else None
            entry = {
                "id": rid, "ts": iso(row["timestamp"]), "text": text,
                "asr": row["asrText"], "formatted": row["formattedText"],
                "edited": edited, "source": "wispr", "app": row["app"],
                "duration_s": round(row["duration"], 2) if row["duration"] else None,
                "words": row["numWords"], "lang": row["detectedLanguage"],
            }
            if edited:
                n_corr += 1
            if export_audio and row["audio"]:
                wav = AUDIO_DIR / f"{rid}.wav"
                wav.write_bytes(row["audio"])
                entry["audio"] = f"audio/{rid}.wav"
                n_audio += 1
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    con.close()
    print(f"history: {n} entries -> {OUT_DIR/'history.jsonl'}")
    print(f"audio: {n_audio} WAVs -> {AUDIO_DIR}")
    print(f"correction pairs (audio+asr+your edit) for fine-tuning: {n_corr}")


if __name__ == "__main__":
    main()
