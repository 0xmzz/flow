"""Turn the imported Wispr corpus into a Whisper fine-tuning dataset.

Reads data/imported-wispr/{history.jsonl, audio/} and produces:
  finetune/dataset/train.jsonl , eval.jsonl   — {audio, text} pairs
  finetune/dataset/stats.json                  — hours, splits, sample rates
  finetune/dataset/corrections.json            — asr→edited substitutions
                                                 (systematic mishears = vocab gold)

Label choice: editedText when you corrected it (high-value hard example),
else asrText (accepted → it was right). formattedText is NOT used — it's
Wispr's LLM formatting, not a verbatim spoken-word target for ASR.
"""

import json
import wave
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "imported-wispr"
OUT = Path(__file__).resolve().parent / "dataset"


def wav_meta(path):
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate(), w.getnframes() / w.getframerate()
    except Exception:
        return None, 0.0


def norm(s):
    return "".join(c.lower() for c in (s or "") if c.isalnum() or c.isspace()).split()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(l) for l in
               (SRC / "history.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    pairs, subs = [], Counter()
    rates = Counter()
    total_dur = corrected = 0.0
    for e in entries:
        if not e.get("audio"):
            continue
        wav = SRC / e["audio"]
        if not wav.exists():
            continue
        edited, asr = e.get("edited"), e.get("asr")
        text = edited or asr or e.get("text")
        if not text or not text.strip():
            continue
        if (e.get("lang") or "en") != "en":
            continue
        sr, dur = wav_meta(wav)
        if dur < 0.4 or dur > 60:
            continue
        rates[sr] += 1
        total_dur += dur
        pairs.append({"audio": str(wav), "text": text.strip(),
                      "corrected": bool(edited)})
        # word-level substitutions asr -> edited (what the model got wrong)
        if edited and asr:
            corrected += dur
            a, b = norm(asr), norm(edited)
            sm = SequenceMatcher(None, a, b)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op in ("replace", "insert") and (j2 - j1) <= 3:
                    frm = " ".join(a[i1:i2]) or "∅"
                    to = " ".join(b[j1:j2])
                    if to:
                        subs[(frm, to)] += 1

    # 90/10 split; keep ALL corrected (hard) examples in train, sample eval
    # from accepted ones so eval reflects normal use and train keeps the signal.
    accepted = [p for p in pairs if not p["corrected"]]
    corrected_p = [p for p in pairs if p["corrected"]]
    n_eval = max(10, len(accepted) // 10)
    eval_set = accepted[:n_eval]
    train_set = corrected_p + accepted[n_eval:]

    def dump(name, rows):
        with (OUT / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"audio": r["audio"], "text": r["text"]},
                                   ensure_ascii=False) + "\n")

    dump("train.jsonl", train_set)
    dump("eval.jsonl", eval_set)

    top_subs = [{"from": k[0], "to": k[1], "count": n}
                for k, n in subs.most_common(40)]
    (OUT / "corrections.json").write_text(
        json.dumps(top_subs, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = {
        "clips": len(pairs),
        "hours": round(total_dur / 3600, 2),
        "corrected_clips": len(corrected_p),
        "corrected_hours": round(corrected / 3600, 2),
        "train": len(train_set), "eval": len(eval_set),
        "sample_rates": dict(rates),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("\nTop asr→correction substitutions (systematic mishears / vocab gold):")
    for s in top_subs[:20]:
        print(f"  {s['count']:2}x  {s['from']!r} -> {s['to']!r}")


if __name__ == "__main__":
    main()
