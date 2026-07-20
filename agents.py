"""Local agent lanes and the single-run background coordinator."""

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DEFAULT_CWD = r"C:\Calude"


class AgentError(RuntimeError):
    outcome = "error"


class AgentCancelled(AgentError):
    outcome = "cancelled"


class AgentTimeout(AgentError):
    outcome = "timeout"


@dataclass
class Lane:
    name: str
    command: tuple
    cwd: str = DEFAULT_CWD
    popen_factory: object = subprocess.Popen
    tree_killer: object = None
    clock: object = time.monotonic
    timeout_s: float = 600.0

    def available(self):
        return bool(self.command and shutil.which(self.command[0]))

    def _kill_tree(self, process):
        if self.tree_killer is not None:
            self.tree_killer(process.pid)
            return
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, creationflags=CREATE_NO_WINDOW, check=False)

    def run(self, task, progress_fn=None, cancel_event=None):
        cancel_event = cancel_event or threading.Event()
        process = self.popen_factory(
            [*self.command, task], cwd=self.cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, creationflags=CREATE_NO_WINDOW)
        lines = []
        output = queue.Queue()

        def read_output():
            try:
                for line in process.stdout:
                    output.put(line)
            finally:
                output.put(None)

        threading.Thread(target=read_output, name=f"flow-{self.name}-stdout",
                         daemon=True).start()
        deadline = self.clock() + self.timeout_s
        reader_done = False
        reason = None
        while True:
            if cancel_event.is_set():
                reason = AgentCancelled("cancelled")
                break
            if self.clock() >= deadline:
                reason = AgentTimeout("timed out after 10 minutes")
                break
            try:
                item = output.get(timeout=0.1)
                if item is None:
                    reader_done = True
                else:
                    clean = item.rstrip("\r\n")
                    lines.append(clean)
                    if clean.strip() and progress_fn:
                        progress_fn(clean.strip()[-80:])
            except queue.Empty:
                pass
            if reader_done and process.poll() is not None:
                break
        if reason is not None:
            self._kill_tree(process)
            try:
                process.wait(timeout=5)
            except Exception:
                pass
            raise reason
        code = process.wait()
        text = "\n".join(lines).strip()
        if code:
            raise AgentError(text[-400:] or f"{self.name} exited with code {code}")
        return text


class HermesLane(Lane):
    def available(self):
        # TODO: replace with a cheap Tailscale endpoint health check.
        return False


LANES = [
    Lane("codex", ("codex", "exec", "--sandbox", "workspace-write")),
    Lane("claude", ("claude", "-p")),
    HermesLane("hermes", ()),
]


def select_lane(name=None, lanes=None):
    lanes = LANES if lanes is None else lanes
    if name:
        selected = next((item for item in lanes if item.name == name), None)
        if selected is None:
            raise AgentError(f"Unknown agent lane: {name}.")
        if not selected.available():
            raise AgentError(f"Agent lane {name} is not available.")
        return selected
    selected = next((item for item in lanes if item.available()), None)
    if selected is None:
        raise AgentError("No agent lanes are available.")
    return selected


def dispatch(task, lane=None, progress_fn=None, cancel_event=None, lanes=None):
    selected = select_lane(lane, lanes)
    return selected.run(task, progress_fn, cancel_event)


def short_summary(text):
    clean = " ".join(str(text or "").split())
    match = re.search(r"[.!?](?:\s|$)", clean)
    first = clean[:match.end()].strip() if match else clean
    return (first or clean)[:160].rstrip()


class AgentRunner:
    """Run at most one lane on a daemon thread and record every started run."""

    def __init__(self, lanes=None, log_path=None, clock=None):
        self.lanes = LANES if lanes is None else lanes
        self.log_path = (Path(log_path) if log_path is not None else
                         Path(__file__).parent / "data" / "agent_runs.jsonl")
        self.clock = clock or time.monotonic
        self.cancel_event = threading.Event()
        self.thread = None
        self._lock = threading.Lock()

    @property
    def running(self):
        return bool(self.thread and self.thread.is_alive())

    def cancel(self):
        self.cancel_event.set()

    def start(self, task, lane=None, progress=None, complete=None):
        with self._lock:
            if self.running:
                return False
            selected = select_lane(lane, self.lanes)
            self.cancel_event = threading.Event()
            self.thread = threading.Thread(
                target=self._run,
                args=(selected, str(task).strip(), progress, complete),
                name="flow-agent", daemon=True)
            self.thread.start()
        return True

    def _run(self, lane, task, progress, complete):
        started = self.clock()
        outcome, result = "ok", ""
        last_progress = [float("-inf")]

        def throttled(line):
            now = self.clock()
            if progress and now - last_progress[0] >= 2.0:
                last_progress[0] = now
                progress(lane.name, line[-80:])

        try:
            result = lane.run(task, throttled, self.cancel_event)
        except AgentError as exc:
            outcome, result = exc.outcome, str(exc)
        except Exception as exc:
            outcome, result = "error", str(exc)
        duration = max(0.0, self.clock() - started)
        self._log(lane.name, task, outcome, duration, result)
        if complete:
            complete({"lane": lane.name, "outcome": outcome, "text": result,
                      "duration_s": duration})

    def _log(self, lane, task, outcome, duration, result):
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lane": lane,
                  "task": task, "outcome": outcome,
                  "duration_s": round(duration, 3),
                  "result_tail": str(result or "")[-400:]}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[agent] log failed: {exc}")
