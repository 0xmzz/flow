"""Hermes session API and local mouthpiece state.

This path is intentionally separate from llm.chain and Flow's dictation store.
"""

import datetime
import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


ERROR = "Hermes is unavailable right now."


class HermesClient:
    def __init__(self, cfg):
        cfg = cfg or {}
        self.url = str(cfg.get("url") or "").rstrip("/")
        self.key = str(cfg.get("key") or "")
        # An agent turn can legitimately take minutes; a session list or health
        # probe never should. One timeout for both would make every cheap
        # lookup able to hang for the length of the slowest possible reply.
        self.timeout = max(1.0, float(cfg.get("timeout_s") or 30))
        self.quick_timeout = max(1.0, min(
            float(cfg.get("quick_timeout_s") or 8), self.timeout))

    def _request(self, method, path, body=None, expected=(200,), quick=False):
        if not self.url or not self.key:
            return {"ok": False, "error": ERROR}
        data = None
        headers = {"Authorization": "Bearer " + self.key,
                   "Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(self.url + path, data=data, headers=headers,
                                     method=method)
        timeout = self.quick_timeout if quick else self.timeout
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                if response.status not in expected:
                    return {"ok": False, "error": ERROR}
            value = json.loads(raw) if raw else {}
            return {"ok": True, "data": value}
        except (OSError, ValueError, urllib.error.HTTPError,
                urllib.error.URLError):
            return {"ok": False, "error": ERROR}

    def health(self):
        return self._request("GET", "/health", quick=True)

    def list_sessions(self, limit=100, offset=0, source=None):
        query = {"limit": max(1, min(int(limit), 500)), "offset": max(0, int(offset))}
        if source:
            query["source"] = source
        return self._request("GET", "/api/sessions?" + urllib.parse.urlencode(query),
                             quick=True)

    def create_session(self, title, session_id=None):
        return self._request(
            "POST", "/api/sessions",
            {"id": session_id or str(uuid.uuid4()), "title": str(title).strip()},
            expected=(200, 201), quick=True)

    def chat(self, session_id, message):
        result = self._request(
            "POST", f"/api/sessions/{urllib.parse.quote(str(session_id), safe='')}/chat",
            {"message": message})
        if not result["ok"]:
            return result
        content = (((result["data"].get("message") or {}).get("content")) or "")
        return {"ok": bool(content), "text": content or ERROR,
                "data": result["data"]}

    def chat_stream(self, session_id, message):
        """Yield decoded SSE data objects; emit one safe error event on failure."""
        if not self.url or not self.key:
            yield {"error": ERROR}
            return
        path = f"/api/sessions/{urllib.parse.quote(str(session_id), safe='')}/chat/stream"
        body = json.dumps({"message": message}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url + path, data=body, method="POST",
            headers={"Authorization": "Bearer " + self.key,
                     "Accept": "text/event-stream",
                     "Content-Type": "application/json; charset=utf-8"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                event = []
                for raw in response:
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if not line:
                        if event:
                            payload = "\n".join(event)
                            event.clear()
                            if payload == "[DONE]":
                                return
                            try:
                                yield json.loads(payload)
                            except ValueError:
                                yield {"text": payload}
                    elif line.startswith("data:"):
                        event.append(line[5:].lstrip())
                if event:
                    payload = "\n".join(event)
                    if payload != "[DONE]":
                        try:
                            yield json.loads(payload)
                        except ValueError:
                            yield {"text": payload}
        except (OSError, ValueError, urllib.error.HTTPError,
                urllib.error.URLError):
            yield {"error": ERROR}

    def start_run(self, task):
        return self._request("POST", "/v1/runs", {"task": task}, expected=(200, 202))

    def run_status(self, run_id):
        return self._request(
            "GET", f"/v1/runs/{urllib.parse.quote(str(run_id), safe='')}")


def session_rows(result):
    if not result.get("ok"):
        return []
    return list((result.get("data") or {}).get("data") or [])


def recent_session(sessions):
    """Return the API's first (most-recent) session, with timestamp fallback."""
    if not sessions:
        return None
    keys = ("last_active_at", "updated_at")
    if any(any(row.get(k) for k in keys) for row in sessions):
        return max(enumerate(sessions), key=lambda item: (
            next((str(item[1].get(k)) for k in keys if item[1].get(k)), ""),
            -item[0]))[1]
    return sessions[0]


class ThreadStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self):
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value.setdefault("threads", {})
                return value
        except (OSError, ValueError):
            pass
        return {"threads": {}, "current": None}

    def save(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self.path)

    def remember(self, title, session_id, current=False):
        value = self.load()
        value["threads"][title] = session_id
        value["active"] = title
        if current:
            value["current"] = title
        self.save(value)

    def clear_current(self):
        value = self.load()
        value["current"] = None
        self.save(value)


class Mouthpiece:
    def __init__(self, cfg, data_dir, client=None):
        self.cfg = cfg or {}
        self.client = client or HermesClient(self.cfg)
        self.data_dir = Path(data_dir)
        self.threads = ThreadStore(self.data_dir / "hermes_threads.json")
        self.turns_path = self.data_dir / "hermes_turns.jsonl"
        self._turn_lock = threading.Lock()
        self._session_locks = {}
        self._locks_lock = threading.Lock()

    def _sessions(self):
        # Drop rows with no usable id here, once, so no caller downstream can
        # KeyError on a malformed API response. A single bad row used to be
        # enough to raise out of resolve() on the hotkey thread.
        return [row for row in session_rows(self.client.list_sessions())
                if isinstance(row, dict) and row.get("id")]

    def predict_title(self, window=""):
        """The thread name a press would land on, resolved with NO network call.

        Rule matching is purely local, so the HUD can name the thread the
        instant the key goes down; the session id is looked up later, off the
        hotkey thread. Falls back to the last thread we actually used.
        """
        for rule in self.cfg.get("thread_rules") or []:
            if not isinstance(rule, dict):
                continue
            match, title = str(rule.get("window") or ""), str(rule.get("title") or "")
            if match and title and match.casefold() in (window or "").casefold():
                return title
        state = self.threads.load()
        return (state.get("current") or state.get("active")
                or self.cfg.get("default_thread_title") or "Hermes")

    @staticmethod
    def _named(sessions, title):
        needle = title.casefold()
        return recent_session([s for s in sessions
                               if str(s.get("title") or "").casefold() == needle])

    def _create(self, title):
        result = self.client.create_session(title)
        if not result.get("ok"):
            return None
        session = (result.get("data") or {}).get("session") or {}
        if not session.get("id"):
            return None
        self.threads.remember(session.get("title") or title, session["id"], current=True)
        return {"id": session["id"], "title": session.get("title") or title}

    def resolve(self, window="", allow_current=True):
        sessions = self._sessions()
        for rule in self.cfg.get("thread_rules") or []:
            if not isinstance(rule, dict):
                continue
            match, title = str(rule.get("window") or ""), str(rule.get("title") or "")
            if match and title and match.casefold() in (window or "").casefold():
                found = self._named(sessions, title)
                if found:
                    self.threads.remember(found["title"], found["id"])
                    return {"id": found["id"], "title": found["title"]}
                cache = self.threads.load()["threads"]
                cached = next(((t, sid) for t, sid in cache.items()
                               if t.casefold() == title.casefold()), None)
                if cached:
                    return {"id": cached[1], "title": cached[0]}
                return self._create(title)
        state = self.threads.load()
        if allow_current and state.get("current"):
            found = self._named(sessions, state["current"])
            if found:
                self.threads.clear_current()
                self.threads.remember(found["title"], found["id"])
                return {"id": found["id"], "title": found["title"]}
        found = recent_session(sessions)
        if found:
            self.threads.remember(found.get("title") or "Hermes", found["id"])
            return {"id": found["id"], "title": found.get("title") or "Hermes"}
        return self._create(self.cfg.get("default_thread_title") or "Flow")

    def local_command(self, text):
        clean = " ".join((text or "").split())
        match = re.match(r"(?i)^new thread\s+(.+)$", clean)
        if match:
            picked = self._create(match.group(1).strip())
            return picked, (f"Created thread {picked['title']}." if picked else ERROR)
        match = re.match(r"(?i)^switch to\s+(.+)$", clean)
        if match:
            title = match.group(1).strip()
            found = self._named(self._sessions(), title)
            if not found:
                return None, f"I couldn't find thread {title}."
            self.threads.remember(found["title"], found["id"], current=True)
            return {"id": found["id"], "title": found["title"]}, \
                f"Switched to {found['title']}."
        if re.match(r"(?i)^what threads[?.!]*$", clean):
            names = [str(s.get("title") or "Untitled") for s in self._sessions()]
            return None, ("Threads: " + ", ".join(names)) if names else "There are no threads."
        return None

    def is_run(self, text):
        clean = (text or "").strip()
        for prefix in self.cfg.get("run_intent_prefixes") or (
                "go build", "kick off", "background task"):
            if clean.casefold().startswith(str(prefix).casefold()):
                task = clean[len(str(prefix)):].lstrip(" ,:—-")
                return task
        return None

    def append_turn(self, thread, session_id, role, text):
        row = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "thread": thread, "session_id": session_id,
               "role": role, "text": text}
        self.turns_path.parent.mkdir(parents=True, exist_ok=True)
        with self._turn_lock, self.turns_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def converse(self, picked, text):
        local = self.local_command(text)
        if local:
            selection, reply = local
            return {"ok": True, "reply": reply, "selection": selection,
                    "local": True}
        task = self.is_run(text)
        if task is not None:
            if not task:
                return {"ok": False, "reply": "Tell me what to build."}
            result = self.client.start_run(task)
            started = result.get("ok") and (result.get("data") or {}).get("run_id")
            return {"ok": bool(started),
                    "reply": ("Started — Hermes will message you on Telegram."
                              if started else ERROR), "run": bool(started)}
        if not picked:
            return {"ok": False, "reply": ERROR}
        self.append_turn(picked["title"], picked["id"], "user", text)
        with self._locks_lock:
            lock = self._session_locks.setdefault(picked["id"], threading.Lock())
        with lock:
            result = self.client.chat(picked["id"], text)
        reply = result.get("text") if result.get("ok") else ERROR
        self.append_turn(picked["title"], picked["id"], "assistant", reply)
        return {"ok": result.get("ok", False), "reply": reply}


def read_turns(path, limit=50):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-max(1, min(int(limit), 500)):]:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except ValueError:
            continue
    return rows
