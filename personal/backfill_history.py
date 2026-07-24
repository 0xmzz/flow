"""One-time: summarize the imported Wispr month into daily briefs for Hermes.

Runs on beast, summarizes locally (the same model the live sweep uses), writes
per-day briefs + a month overview to data/imported-wispr/briefs/. A companion
step scp's them to Hermes's imports/ and the vault. No audio, text only.
"""

import collections
import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
import llm

cfg = json.loads((APP / "config.json").read_text(encoding="utf-8-sig"))
SRC = APP / "data" / "imported-wispr" / "history.jsonl"
OUT = APP / "data" / "imported-wispr" / "briefs"
OUT.mkdir(parents=True, exist_ok=True)

DAY_SYS = ("You maintain a personal AI assistant's memory of the user's days. "
           "Given the user's voice activity for one day (dictations across apps), "
           "write a tight brief: one-line summary then 3-6 terse bullets of "
           "themes/threads/decisions. Under 130 words. No preamble.")

entries = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
by_day = collections.defaultdict(list)
for e in entries:
    day = (e.get("ts") or "")[:10]
    if day and e.get("text"):
        by_day[day].append(e)

briefs = {}
for day in sorted(by_day):
    rows = by_day[day]
    lines = [f"- [{r.get('app', '?')}] {r['text'].strip()}" for r in rows]
    body = f"Voice activity for {day}:\n" + "\n".join(lines)
    brief, lane = llm.chain(cfg["cleanup"], DAY_SYS, body[:12000], max_tokens=400)
    if brief:
        (OUT / f"flow-context-{day}.md").write_text(
            f"# Flow context — {day} (backfilled from Wispr)\n\n{brief}\n",
            encoding="utf-8")
        briefs[day] = brief
        print(f"{day}: {len(rows)} entries -> brief ({lane})")

overview, _ = llm.chain(
    cfg["cleanup"],
    "Summarize this month of daily voice-activity briefs into a compact overview "
    "of the user's recurring themes, ongoing projects, key decisions, and open "
    "concerns. Under 250 words. No preamble.",
    "\n\n".join(f"{d}:\n{b}" for d, b in sorted(briefs.items()))[:14000],
    max_tokens=600)
if overview:
    (OUT / "wispr-history-overview.md").write_text(
        f"# Flow — voice history overview (Jun–Jul 2026, backfilled from Wispr)\n\n"
        f"{overview}\n", encoding="utf-8")
    print(f"\noverview written: {len(overview)} chars")
print(f"\n{len(briefs)} daily briefs -> {OUT}")
