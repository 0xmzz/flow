import json
from datetime import datetime, timezone
from pathlib import Path

import export as flow_export


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _cfg(dest, **export_patch):
    settings = {
        "enabled": True, "dest": str(dest), "mode": "folder",
        "git_remote": "",
        "include": {"history": True, "workflows": True, "intents": True,
                    "audio": False},
        "interval_min": 60, "redact": [],
    }
    settings.update(export_patch)
    return {"history": {"dir": "data"}, "export": settings}


def _entry(entry_id, ts, text, **extra):
    row = {"id": entry_id, "ts": ts, "text": text, "cleaned": text,
           "source": "local", "window": "Editor"}
    row.update(extra)
    return row


def test_private_rows_never_reach_files_or_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(flow_export, "APP_DIR", tmp_path)
    dest = tmp_path / "out"
    _write_jsonl(tmp_path / "data" / "history.jsonl", [
        _entry("public", "2026-07-01T10:00:00+00:00", "visible"),
        _entry("private", "2026-07-01T11:00:00+00:00",
               "secret phrase", private=True),
    ])
    _write_jsonl(tmp_path / "data" / "intent_log.jsonl", [
        {"ts": "2026-07-01T10:00:00", "text": "visible command",
         "action": "open", "source": "fastpath"},
        {"ts": "2026-07-01T11:00:00", "text": "secret command",
         "action": "open", "source": "fastpath", "private": True},
    ])

    summary = flow_export.export_now(_cfg(dest))

    exported = "\n".join(
        path.read_text(encoding="utf-8") for path in dest.rglob("*")
        if path.is_file())
    assert "secret phrase" not in exported
    assert "secret command" not in exported
    assert summary["entries"] == 1
    assert json.loads((dest / "manifest.json").read_text())["counts"]["history"] == 1
    assert "Total entries: 1" in (
        dest / "CONTEXT.md").read_text(encoding="utf-8")


def test_redacts_text_and_cleaned_and_context_needs_no_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(flow_export, "APP_DIR", tmp_path)
    dest = tmp_path / "out"
    _write_jsonl(tmp_path / "data" / "history.jsonl", [
        _entry("one", "2026-07-01T10:00:00+00:00", "token 123",
               cleaned="clean token 456"),
    ])
    monkeypatch.setitem(
        __import__("sys").modules, "llm",
        type("ExplodingLLM", (), {"__getattr__": lambda *_: (_ for _ in ()).throw(
            AssertionError("LLM must not be called"))})())

    summary = flow_export.export_now(
        _cfg(dest, redact=[r"\d+", "token"]))

    shard = next((dest / "history").glob("*.jsonl")).read_text()
    assert "123" not in shard and "456" not in shard and "token" not in shard
    assert shard.count("[redacted]") == 4
    assert summary["redacted_spans"] == 4
    assert (dest / "CONTEXT.md").is_file()


def test_incremental_rewrites_only_changed_day_and_preserves_user_file(
        tmp_path, monkeypatch):
    monkeypatch.setattr(flow_export, "APP_DIR", tmp_path)
    dest = tmp_path / "out"
    history = tmp_path / "data" / "history.jsonl"
    rows = [
        _entry("one", "2026-07-01T10:00:00+00:00", "first"),
        _entry("two", "2026-07-02T10:00:00+00:00", "second"),
    ]
    _write_jsonl(history, rows)
    flow_export.export_now(_cfg(dest))
    shards = sorted((dest / "history").glob("*.jsonl"))
    before = {path.name: (path.stat().st_mtime_ns, path.read_bytes())
              for path in shards}
    user_file = dest / "notes.md"
    user_file.write_text("keep me", encoding="utf-8")

    rows.append(_entry("late", "2026-07-01T08:00:00+00:00", "late old day"))
    _write_jsonl(history, rows)
    summary = flow_export.export_now(_cfg(dest))
    after = {path.name: (path.stat().st_mtime_ns, path.read_bytes())
             for path in shards}

    assert summary["changed_days"] == 1
    changed = [name for name in before if before[name] != after[name]]
    assert len(changed) == 1
    assert "late old day" in after[changed[0]][1].decode()
    assert user_file.read_text(encoding="utf-8") == "keep me"

    stable = flow_export.export_now(_cfg(dest))
    assert not stable["changed"]


def test_audio_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(flow_export, "APP_DIR", tmp_path)
    dest = tmp_path / "out"
    wav = tmp_path / "data" / "dataset" / "2026-07-01" / "one.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"RIFFfake")
    _write_jsonl(tmp_path / "data" / "history.jsonl", [
        _entry("one", "2026-07-01T10:00:00+00:00", "audio row",
               audio="dataset/2026-07-01/one.wav"),
    ])

    flow_export.export_now(_cfg(dest))
    assert not (dest / "audio").exists()

    flow_export.export_now(_cfg(
        dest, include={"history": True, "workflows": True, "intents": True,
                       "audio": True}))
    assert next((dest / "audio").rglob("*.wav")).read_bytes() == b"RIFFfake"


class _Result:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def test_git_commits_only_on_change_and_does_not_push_without_remote(
        tmp_path, monkeypatch):
    monkeypatch.setattr(flow_export, "APP_DIR", tmp_path)
    dest = tmp_path / "repo"
    dest.mkdir()
    _write_jsonl(tmp_path / "data" / "history.jsonl", [
        _entry("one", "2026-07-01T10:00:00+00:00", "first"),
    ])
    calls = []

    def runner(args, **kwargs):
        calls.append((list(args), kwargs))
        if args[1:3] == ["rev-parse", "--is-inside-work-tree"]:
            return _Result("true\n")
        if args[1:3] == ["rev-parse", "--git-path"]:
            return _Result(str(dest / (".git-" + args[-1])) + "\n")
        if args[1:3] == ["symbolic-ref", "--quiet"]:
            return _Result("main\n")
        if args[1:4] == ["diff", "--cached", "--quiet"]:
            return _Result(returncode=1)
        return _Result()

    cfg = _cfg(dest, mode="git")
    first = flow_export.export_now(cfg, runner=runner)
    assert first["committed"]
    assert any(call[0][1:3] == ["add", "-A"] for call in calls)
    assert any(call[0][1] == "commit" for call in calls)
    assert not any(call[0][1] == "push" for call in calls)
    assert all(call[1]["cwd"] == str(dest) for call in calls)
    assert all(call[1]["creationflags"] == flow_export.CREATE_NO_WINDOW
               for call in calls)

    calls.clear()
    second = flow_export.export_now(cfg, runner=runner)
    assert not second["changed"]
    assert not any(call[0][1] in {"add", "commit", "push"} for call in calls)


def test_voice_export_is_a_fast_path(tmp_path):
    from commands import Router

    called = []
    router = Router(
        {"cleanup": {}, "local_only": True},
        intent_log_path=tmp_path / "intent_log.jsonl",
        export_fn=lambda: called.append(True) or
        {"entries": 3, "dest": str(tmp_path / "my-agent-data")})
    response = router.handle("flow, export my data")
    assert response == "Exported 3 entries to my-agent-data."
    assert called == [True]
