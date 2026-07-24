"""User-directed, agent-readable export of Flow's local voice data."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


APP_DIR = Path(__file__).parent
FLOW_VERSION = "1.0"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_export_lock = threading.Lock()


def _json_bytes(value, *, pretty=False):
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return (text + "\n").encode("utf-8")


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return False
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return True


def _compile_redactions(patterns):
    compiled = []
    for pattern in patterns or ():
        try:
            compiled.append(re.compile(str(pattern)))
        except re.error as exc:
            raise ValueError(f"invalid export redaction regex {pattern!r}: {exc}") from exc
    return compiled


def _redact_row(row, patterns):
    clean = dict(row)
    count = 0
    for field in ("text", "cleaned"):
        value = clean.get(field)
        if not isinstance(value, str):
            continue
        for pattern in patterns:
            value, matches = pattern.subn("[redacted]", value)
            count += matches
        clean[field] = value
    return clean, count


def _local_datetime(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone()
    except (TypeError, ValueError):
        return None


def _fingerprint(rows):
    return hashlib.sha256(b"".join(_json_bytes(row) for row in rows)).hexdigest()


def _git_call(runner, args, dest, *, check=True):
    return runner(
        args, cwd=str(dest), capture_output=True, text=True, check=check,
        creationflags=CREATE_NO_WINDOW)


def _git_preflight(dest, runner):
    if not dest.is_dir():
        raise ValueError(f"git export destination does not exist: {dest}")
    inside = _git_call(
        runner, ["git", "rev-parse", "--is-inside-work-tree"], dest)
    if inside.stdout.strip() != "true":
        raise ValueError(f"export destination is not a git worktree: {dest}")
    for name in ("rebase-merge", "rebase-apply"):
        result = _git_call(
            runner, ["git", "rev-parse", "--git-path", name], dest)
        path = Path(result.stdout.strip())
        if not path.is_absolute():
            path = dest / path
        if path.exists():
            raise RuntimeError(f"git {name} in progress; export aborted")
    unmerged = _git_call(runner, ["git", "ls-files", "-u"], dest)
    if unmerged.stdout.strip():
        raise RuntimeError("git conflicts present; export aborted")
    branch = _git_call(
        runner, ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        dest, check=False)
    if getattr(branch, "returncode", 0):
        raise RuntimeError("git worktree has detached HEAD; export aborted")


def _context(entries, intents, workflows):
    dated = [(row, _local_datetime(row.get("ts"))) for row in entries]
    dated = [(row, value) for row, value in dated if value is not None]
    hours = Counter(value.hour for _row, value in dated)
    apps = Counter(str(row.get("window")) for row in entries if row.get("window"))
    workflow_runs = Counter(
        str(row.get("arg")) for row in intents
        if row.get("action") == "workflow" and row.get("arg"))

    def ranked(counter, formatter=str, limit=5):
        values = sorted(counter.items(), key=lambda item: (-item[1], str(item[0]).casefold()))
        return ", ".join(f"{formatter(key)} ({count})" for key, count in values[:limit]) or "none"

    dates = sorted(value.date().isoformat() for _row, value in dated)
    date_range = f"{dates[0]} to {dates[-1]}" if dates else "none"
    workflow_names = {str(row.get("name")) for row in workflows if row.get("name")}
    unused = len(workflow_names - set(workflow_runs))
    hour_label = lambda hour: f"{hour:02d}:00"
    return f"""# Flow voice-data context

This directory is an agent-readable export of the user's voice data from Flow. Timestamps are grouped and analyzed in the machine's local time; row `ts` values retain their original ISO-8601 timezone.

## File layout

- `CONTEXT.md` — this entry point, schemas, statistics, and usage guidance.
- `history/YYYY-MM-DD.jsonl` — non-private dictations grouped by local calendar day.
- `workflows/<name>.json` — the user's taught Flow workflow recipes.
- `intents/intent_log.jsonl` — non-private utterance-to-intent routing examples.
- `audio/YYYY-MM-DD/<id>.wav` — optional paired audio, present only when explicitly enabled.
- `manifest.json` — generation time, counts, incremental cursor, and Flow version.

## Schemas

History row:
```json
{{"id":"string","ts":"ISO-8601","text":"raw transcript","cleaned":"pasted text or null","private":null,"source":"string","window":"string or null","duration_s":0.0,"asr_s":0.0,"words":0,"model":"string","device":"string","audio":"relative source path (optional)"}}
```

Intent row:
```json
{{"ts":"ISO-8601","text":"utterance","action":"resolved action","arg":"optional argument","source":"fastpath or llm","context_chars":0}}
```

## Rolling stats

- Total entries: {len(entries)}
- Date range: {date_range}
- Busiest hours: {ranked(hours, hour_label)}
- Top apps dictated into: {ranked(apps)}
- Most-used workflows: {ranked(workflow_runs)}
- Taught workflows: {len(workflow_names)} ({unused} not represented in the intent log)

## How to use this

An agent can search the daily history shards to answer “What did I say about X?”, analyze repeated topics or wording to answer “What do I dictate most?”, compare raw and cleaned text, identify the apps that receive the most dictation, and inspect the intent log plus recipes to answer “Which workflows do I actually run?”
"""


def _normalized_export(cfg):
    defaults = {
        "enabled": False, "dest": "", "mode": "folder", "git_remote": "",
        "include": {"history": True, "workflows": True, "intents": True,
                    "audio": False},
        "interval_min": 60, "redact": [],
    }
    supplied = cfg.get("export") or {}
    result = dict(defaults)
    result.update({k: v for k, v in supplied.items() if k != "include"})
    result["include"] = {**defaults["include"], **(supplied.get("include") or {})}
    return result


def export_now(cfg, *, runner=subprocess.run, now_fn=None):
    """Export configured Flow data and return a compact run summary."""
    settings = _normalized_export(cfg)
    raw_dest = str(settings.get("dest") or "")
    dest = Path(raw_dest)
    if not raw_dest or not dest.is_absolute():
        raise ValueError("export.dest must be a non-empty absolute path")
    if settings["mode"] not in {"folder", "git"}:
        raise ValueError("export.mode must be 'folder' or 'git'")
    patterns = _compile_redactions(settings.get("redact"))
    now = (now_fn or (lambda: datetime.now(timezone.utc)))()

    with _export_lock:
        if settings["mode"] == "git":
            _git_preflight(dest, runner)
        else:
            dest.mkdir(parents=True, exist_ok=True)

        manifest_path = dest / "manifest.json"
        previous = {}
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        old_cursor = previous.get("cursor") if isinstance(previous.get("cursor"), dict) else {}
        old_hashes = old_cursor.get("days") if isinstance(old_cursor.get("days"), dict) else {}
        old_ids = set(old_cursor.get("history_ids") or ())

        history_dir = APP_DIR / (cfg.get("history") or {}).get("dir", "data")
        history = [row for row in _read_jsonl(history_dir / "history.jsonl")
                   if not row.get("private")]
        intents_path = history_dir / "intent_log.jsonl"
        intents = [row for row in _read_jsonl(intents_path) if not row.get("private")]
        workflows_dir = history_dir / "workflows"
        workflows = []
        workflow_files = []
        if workflows_dir.exists():
            for path in sorted(workflows_dir.glob("*.json")):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    workflows.append(value)
                    workflow_files.append((path.name, value))

        redactions = 0
        redacted_history = []
        for row in history:
            clean, count = _redact_row(row, patterns)
            redacted_history.append(clean)
            redactions += count
        redacted_intents = []
        for row in intents:
            clean, count = _redact_row(row, patterns)
            redacted_intents.append(clean)
            redactions += count

        days = {}
        for row in redacted_history:
            local = _local_datetime(row.get("ts"))
            day = local.date().isoformat() if local else "unknown"
            days.setdefault(day, []).append(row)
        day_hashes = {day: _fingerprint(rows) for day, rows in days.items()}
        changed_days = [
            day for day, digest in day_hashes.items()
            if (old_hashes.get(day) != digest
                or (settings["include"]["history"]
                    and not (dest / "history" / f"{day}.jsonl").exists()))
        ]
        changed = False
        if settings["include"]["history"]:
            for day in changed_days:
                changed |= _atomic_write(
                    dest / "history" / f"{day}.jsonl",
                    b"".join(_json_bytes(row) for row in days[day]))

        if settings["include"]["workflows"]:
            for name, value in workflow_files:
                changed |= _atomic_write(
                    dest / "workflows" / name, _json_bytes(value, pretty=True))
        if settings["include"]["intents"]:
            changed |= _atomic_write(
                dest / "intents" / "intent_log.jsonl",
                b"".join(_json_bytes(row) for row in redacted_intents))

        audio_files = audio_bytes = 0
        if settings["include"]["audio"]:
            for row in redacted_history:
                relative = row.get("audio")
                local = _local_datetime(row.get("ts"))
                if not relative or local is None:
                    continue
                source = (history_dir / relative).resolve()
                try:
                    source.relative_to(history_dir.resolve())
                except ValueError:
                    print(f"[export] skipped unsafe audio path: {relative}")
                    continue
                if not source.is_file():
                    print(f"[export] missing audio: {relative}")
                    continue
                target = dest / "audio" / local.date().isoformat() / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.read_bytes() != source.read_bytes():
                    shutil.copy2(source, target)
                    changed = True
                audio_files += 1
                audio_bytes += source.stat().st_size
            print(f"[export] AUDIO ENABLED: {audio_files} wav files, "
                  f"{audio_bytes / (1024 * 1024):.1f} MiB")

        context_data = _context(
            redacted_history if settings["include"]["history"] else [],
            redacted_intents if settings["include"]["intents"] else [],
            workflows if settings["include"]["workflows"] else [])
        changed |= _atomic_write(dest / "CONTEXT.md", context_data.encode("utf-8"))

        current_ids = [str(row.get("id")) for row in history if row.get("id")]
        new_entries = sum(value not in old_ids for value in current_ids)
        counts = {"history": len(history), "workflows": len(workflows),
                  "intents": len(intents), "audio": audio_files,
                  "redacted_spans": redactions}
        cursor = {"version": 1, "days": day_hashes, "history_ids": current_ids}
        material = (changed or previous.get("counts") != counts
                    or old_cursor != cursor
                    or previous.get("flow_version") != FLOW_VERSION)
        generated_at = (now.isoformat(timespec="seconds") if material
                        else previous.get("generated_at",
                                          now.isoformat(timespec="seconds")))
        manifest = {"generated_at": generated_at, "counts": counts,
                    "cursor": cursor, "flow_version": FLOW_VERSION}
        manifest_changed = _atomic_write(
            manifest_path, _json_bytes(manifest, pretty=True))
        changed = changed or manifest_changed

        committed = False
        pushed = False
        if settings["mode"] == "git" and changed:
            _git_call(runner, ["git", "add", "-A"], dest)
            staged = _git_call(
                runner, ["git", "diff", "--cached", "--quiet"], dest,
                check=False)
            if getattr(staged, "returncode", 0):
                _git_call(
                    runner, ["git", "commit", "-m",
                             f"flow: {new_entries} new entries"], dest)
                committed = True
                if settings.get("git_remote"):
                    _git_call(
                        runner, ["git", "push", str(settings["git_remote"])],
                        dest)
                    pushed = True

        print(f"[export] {len(history)} entries, {len(changed_days)} changed days, "
              f"{redactions} redacted spans -> {dest}")
        return {"entries": len(history), "new_entries": new_entries,
                "changed_days": len(changed_days), "redacted_spans": redactions,
                "audio_files": audio_files, "dest": str(dest),
                "changed": changed, "committed": committed, "pushed": pushed,
                "counts": counts}
