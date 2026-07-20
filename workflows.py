"""Persistent, replayable voice workflows.

This module deliberately contains no HUD or live-listening code.  Both the
store and runner accept their environment as arguments, which keeps workflow
tests (and future front ends) independent of Windows automation.
"""

from dataclasses import dataclass
import difflib
import json
from pathlib import Path
import re
import threading
import time


PRIMITIVES = frozenset({
    "open", "close", "switch", "price", "search", "type", "keys",
    "wait_s", "wait_window", "say", "ask",
})


@dataclass
class Question:
    """A prompt whose handler consumes the next spoken answer."""

    text: str
    handler: object
    cancel: object = None

    @property
    def question(self):
        """Compatibility name used by UI code."""
        return self.text

    @property
    def cancel_callback(self):
        return self.cancel


def normalize(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _validate_step(step):
    if not isinstance(step, dict) or step.get("do") not in PRIMITIVES:
        return False
    if "arg" not in step:
        return False
    if "timeout_s" in step and not isinstance(step["timeout_s"], (int, float)):
        return False
    if "yes" in step:
        if step["do"] != "ask" or not isinstance(step["yes"], list):
            return False
        if not all(_validate_step(child) for child in step["yes"]):
            return False
    return True


def validate_recipe(recipe):
    return (isinstance(recipe, dict)
            and isinstance(recipe.get("name"), str)
            and bool(recipe["name"].strip())
            and isinstance(recipe.get("triggers"), list)
            and bool(recipe["triggers"])
            and all(isinstance(t, str) and t.strip() for t in recipe["triggers"])
            and isinstance(recipe.get("steps"), list)
            and all(_validate_step(step) for step in recipe["steps"]))


class RecipeStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path(__file__).parent / "data" / "workflows"
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _filename(name):
        safe = re.sub(r"[^a-z0-9]+", "-", normalize(name)).strip("-")
        if not safe:
            raise ValueError("workflow name has no usable characters")
        return safe + ".json"

    def _load_all(self):
        recipes = []
        for path in sorted(self.root.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if validate_recipe(value):
                recipes.append(value)
        return recipes

    def list(self):
        return self._load_all()

    def names(self):
        return [recipe["name"] for recipe in self._load_all()]

    def get(self, name):
        wanted = normalize(name)
        return next((r for r in self._load_all()
                     if normalize(r["name"]) == wanted), None)

    def match(self, text):
        heard = normalize(text)
        best = (0.0, None)
        for recipe in self._load_all():
            for trigger in recipe["triggers"]:
                score = difflib.SequenceMatcher(None, heard, normalize(trigger)).ratio()
                if score > best[0]:
                    best = (score, recipe)
        return best[1] if best[0] >= 0.85 else None

    def save(self, recipe):
        if not validate_recipe(recipe):
            raise ValueError("invalid workflow recipe")
        # JSON round-trip also detaches the stored value from caller mutation.
        clean = json.loads(json.dumps(recipe))
        path = self.root / self._filename(clean["name"])
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return clean

    def delete(self, name):
        recipe = self.get(name)
        if recipe is None:
            return False
        try:
            (self.root / self._filename(recipe["name"])).unlink()
        except FileNotFoundError:
            return False
        return True


class WorkflowRunner:
    """Execute one recipe at a time on a daemon worker."""

    def __init__(self, router=None, primitives=None, key_sender=None,
                 title_supplier=None, wait_fn=None, clock=None, speak_fn=None,
                 ask_fn=None):
        self.router = router
        self.primitives = primitives or {}
        self.key_sender = key_sender or self._default_keys
        self.title_supplier = title_supplier or self._default_titles
        self.wait_fn = wait_fn
        self.clock = clock or time.monotonic
        self.speak_fn = speak_fn or (lambda _text: None)
        self.ask_fn = ask_fn
        self.cancel_event = threading.Event()
        self.thread = None
        self._lock = threading.Lock()

    @staticmethod
    def _default_keys(keys):
        import keyboard
        keyboard.send(keys)

    @staticmethod
    def _default_titles():
        try:
            from commands import _windows_list
            return [title for _hwnd, title in _windows_list()]
        except Exception:
            return []

    @property
    def running(self):
        return bool(self.thread and self.thread.is_alive())

    def cancel(self):
        self.cancel_event.set()

    def start(self, recipe, progress=None, complete=None):
        if not validate_recipe(recipe):
            raise ValueError("invalid workflow recipe")
        with self._lock:
            if self.running:
                return False
            self.cancel_event = threading.Event()
            self.thread = threading.Thread(
                target=self._run, args=(recipe, progress, complete),
                name="flow-workflow", daemon=True)
            self.thread.start()
        return True

    run = start

    def _run(self, recipe, progress, complete):
        result = f"Workflow {recipe['name']} complete."
        try:
            total = len(recipe["steps"])
            for index, step in enumerate(recipe["steps"], 1):
                if self.cancel_event.is_set():
                    result = f"Workflow {recipe['name']} cancelled."
                    break
                if progress:
                    progress(index, total, self.describe(step))
                self._execute(step)
            if self.cancel_event.is_set():
                result = f"Workflow {recipe['name']} cancelled."
        except Exception as exc:
            result = f"Workflow {recipe['name']} failed: {exc}"
        if complete:
            complete(result)

    @staticmethod
    def describe(step):
        action = step["do"].replace("_", " ")
        return f"{action}: {step.get('arg', '')}".rstrip(": ")

    def _pause(self, seconds):
        deadline = self.clock() + max(0.0, float(seconds))
        while not self.cancel_event.is_set():
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            delay = min(0.1, remaining)
            if self.wait_fn:
                self.wait_fn(delay)
            else:
                self.cancel_event.wait(delay)

    def _router_call(self, action, arg):
        if callable(self.primitives):
            return self.primitives(action, arg)
        fn = self.primitives.get(action)
        if fn is None and self.router is not None:
            fn = getattr(self.router, "_" + action, None)
            if fn is None:
                fn = getattr(self.router, action, None)
        if fn is None:
            raise RuntimeError(f"workflow primitive unavailable: {action}")
        return fn(arg)

    def _execute(self, step):
        action, arg = step["do"], step.get("arg")
        if action in {"open", "close", "switch", "price", "search", "type"}:
            self._router_call(action, arg)
        elif action == "keys":
            self.key_sender(str(arg))
        elif action == "wait_s":
            self._pause(min(30.0, max(0.0, float(arg))))
        elif action == "wait_window":
            timeout = min(30.0, max(0.0, float(step.get("timeout_s", 10))))
            deadline = self.clock() + timeout
            wanted = normalize(arg)
            while not self.cancel_event.is_set():
                titles = [normalize(title) for title in self.title_supplier()]
                if any(wanted in title or difflib.SequenceMatcher(
                        None, wanted, title).ratio() >= 0.7 for title in titles):
                    return
                if self.clock() >= deadline:
                    raise TimeoutError(f"window did not appear: {arg}")
                self._pause(min(0.1, deadline - self.clock()))
        elif action == "say":
            self.speak_fn(str(arg))
        elif action == "ask":
            if self.ask_fn is None:
                raise RuntimeError("workflow question bridge unavailable")
            answer = self.ask_fn(str(arg), self.cancel_event)
            if answer is True:
                for child in step.get("yes", []):
                    if self.cancel_event.is_set():
                        break
                    self._execute(child)


def creation_dialog(name, store, intent_parser):
    """Build the first Question in a side-effect-free voice creation dialog.

    ``intent_parser(answer)`` may return ``(action, arg)`` or a dict with
    ``action``/``do`` and ``arg``. Only workflow-safe primitives are accepted.
    """
    workflow_name = str(name).strip()
    steps = []
    failures = 0

    def parse(answer):
        value = intent_parser(answer)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            action, arg = value[0], value[1]
        elif isinstance(value, dict):
            action, arg = value.get("do", value.get("action")), value.get("arg")
        else:
            return None
        if action not in PRIMITIVES or action == "ask" or arg is None:
            return None
        step = {"do": action, "arg": arg}
        return step if _validate_step(step) else None

    def handle(answer):
        nonlocal failures
        heard = normalize(answer)
        if heard in {"cancel", "never mind", "nevermind"}:
            return "Workflow creation cancelled."
        if heard == "done":
            if not steps:
                return "Workflow creation cancelled; no steps were added."
            recipe = {"name": workflow_name,
                      "triggers": [normalize(workflow_name)], "steps": steps}
            store.save(recipe)
            return f"Saved workflow {workflow_name} with {len(steps)} steps."
        step = parse(answer)
        if step is None:
            failures += 1
            if failures >= 2:
                return "I couldn't understand that step. Workflow creation cancelled."
            return Question("I didn't understand that step. Please try once more, or say done.", handle)
        failures = 0
        steps.append(step)
        return Question(f"Added {WorkflowRunner.describe(step)}. What is the next step? Say done when finished.", handle)

    return Question(f"Creating workflow {workflow_name}. What is the first step?", handle)


# Friendly aliases for callers that prefer builder terminology.
build_creation_dialog = creation_dialog
