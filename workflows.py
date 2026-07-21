"""Persistent, replayable voice workflows.

This module deliberately contains no HUD or live-listening code.  Both the
store and runner accept their environment as arguments, which keeps workflow
tests (and future front ends) independent of Windows automation.

Recipes are linear step lists. ``if_window`` may contain JSON-authored
``then`` and ``else`` lists; voice creation currently teaches one ``then``
step only. There are deliberately no joins or parallel branches.
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
    "wait_s", "wait_window", "say", "ask", "place", "place_rect", "desktop",
    "agent", "run", "if_window",
})

PLACE_REGIONS = frozenset({
    "full", "left", "right", "top", "bottom",
    "top-left", "top-right", "bottom-left", "bottom-right",
})


@dataclass
class Question:
    """A prompt whose handler consumes the next spoken answer."""

    text: str
    handler: object
    cancel: object = None
    cancel_text: str = "Cancelled."

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
    if step["do"] == "place":
        if not isinstance(step["arg"], str) or not step["arg"].strip():
            return False
        if step.get("region", "full") not in PLACE_REGIONS:
            return False
        monitor = step.get("monitor")
        if monitor is not None and (not isinstance(monitor, int) or monitor < 1):
            return False
    if step["do"] == "place_rect":
        if not isinstance(step["arg"], str) or not step["arg"].strip():
            return False
        values = [step.get(key) for key in ("monitor", "x", "y", "w", "h")]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            return False
        if step["monitor"] < 1 or step["w"] < 1 or step["h"] < 1:
            return False
    if step["do"] == "desktop" and step["arg"] not in {"next", "previous"}:
        return False
    if step["do"] in {"agent", "run", "if_window"}:
        if not isinstance(step["arg"], str) or not step["arg"].strip():
            return False
    if step["do"] == "agent" and step.get("lane") not in {None, "codex", "claude"}:
        return False
    if step["do"] == "if_window":
        if not isinstance(step.get("then"), list):
            return False
        if not all(_validate_step(child) for child in step["then"]):
            return False
        if "else" in step and (not isinstance(step["else"], list)
                               or not all(_validate_step(child) for child in step["else"])):
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
                 ask_fn=None, recipe_store=None, agent_runner=None):
        self.router = router
        self.primitives = primitives or {}
        self.key_sender = key_sender or self._default_keys
        self.title_supplier = title_supplier or self._default_titles
        self.wait_fn = wait_fn
        self.clock = clock or time.monotonic
        self.speak_fn = speak_fn or (lambda _text: None)
        self.ask_fn = ask_fn
        self.recipe_store = recipe_store
        self.agent_runner = agent_runner
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
            self._execute_recipe(recipe, progress, [recipe["name"]])
            if self.cancel_event.is_set():
                result = f"Workflow {recipe['name']} cancelled."
        except Exception as exc:
            result = f"Workflow {recipe['name']} failed: {exc}"
        if complete:
            complete(result)

    @staticmethod
    def describe(step):
        action = step["do"].replace("_", " ")
        return f"{action} {step.get('arg', '')}".rstrip()

    def _execute_recipe(self, recipe, progress, stack):
        total = len(recipe["steps"])
        prefix = " > ".join(stack[1:])
        for index, step in enumerate(recipe["steps"], 1):
            if self.cancel_event.is_set():
                return
            action = step["do"]
            if action == "run":
                self._execute_run(step, progress, stack)
                continue
            description = self.describe(step)
            shown = f"{prefix} > {description}" if prefix else description
            if progress:
                progress(index, total, shown)
            self._execute(step, progress=progress, context=(index, total, prefix),
                          source_name=recipe["name"], stack=stack)

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

    def _window_matches(self, query):
        wanted = normalize(query)
        titles = [normalize(title) for title in self.title_supplier()]
        return any(wanted in title or difflib.SequenceMatcher(
            None, wanted, title).ratio() >= 0.7 for title in titles)

    def _execute_run(self, step, progress, stack):
        if self.recipe_store is None:
            raise RuntimeError("workflow store unavailable for nested run")
        recipe = self.recipe_store.get(step["arg"])
        if recipe is None:
            raise RuntimeError(f"workflow not found: {step['arg']}")
        names = [normalize(name) for name in stack]
        target = normalize(recipe["name"])
        if target in names:
            chain = " > ".join(stack + [recipe["name"]])
            raise RuntimeError(f"recursive workflow: {chain}")
        if len(stack) >= 3:
            raise RuntimeError("workflow nesting exceeds maximum depth 3")
        self._execute_recipe(recipe, progress, stack + [recipe["name"]])

    def _execute_agent(self, step, progress, context, source_name):
        if self.agent_runner is None:
            raise RuntimeError("agent runner unavailable")
        completed = threading.Event()
        result = {}
        index, total, prefix = context

        def on_progress(lane, tail):
            description = f"{lane} — {tail}"
            if prefix:
                description = f"{prefix} > {description}"
            if progress:
                progress(index, total, description)

        def on_complete(value):
            result.update(value)
            completed.set()

        started = self.agent_runner.start(
            step["arg"], step.get("lane"), progress=on_progress,
            complete=on_complete, source=f"workflow:{source_name}")
        if not started:
            raise RuntimeError("an agent is already running")
        cancellation_sent = False
        while not completed.wait(0.1):
            if self.cancel_event.is_set() and not cancellation_sent:
                self.agent_runner.cancel()
                cancellation_sent = True
        outcome = result.get("outcome", "error")
        if outcome == "cancelled":
            self.cancel_event.set()
        if outcome != "ok" and not (outcome == "cancelled" and self.cancel_event.is_set()):
            message = " ".join(str(result.get("text") or outcome).split())
            raise RuntimeError(f"agent {outcome}: {message[:160]}")

    def _execute(self, step, progress=None, context=None, source_name=None,
                 stack=None):
        action, arg = step["do"], step.get("arg")
        if action in {"open", "close", "switch", "price", "search", "type",
                      "desktop"}:
            self._router_call(action, arg)
        elif action in {"place", "place_rect"}:
            if self.router is not None:
                getattr(self.router, "_" + action + "_step")(step)
            elif callable(self.primitives):
                self.primitives(action, dict(step))
            else:
                fn = self.primitives.get(action)
                if fn is None:
                    raise RuntimeError(f"workflow primitive unavailable: {action}")
                fn(dict(step))
        elif action == "keys":
            self.key_sender(str(arg))
        elif action == "wait_s":
            self._pause(min(30.0, max(0.0, float(arg))))
        elif action == "wait_window":
            timeout = min(30.0, max(0.0, float(step.get("timeout_s", 10))))
            deadline = self.clock() + timeout
            wanted = normalize(arg)
            while not self.cancel_event.is_set():
                if self._window_matches(wanted):
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
                    self._execute(child, progress=progress, context=context,
                                  source_name=source_name, stack=stack)
        elif action == "agent":
            self._execute_agent(step, progress, context, source_name)
        elif action == "run":
            self._execute_run(step, progress, stack)
        elif action == "if_window":
            branch = step["then"] if self._window_matches(arg) else step.get("else", [])
            for child in branch:
                if self.cancel_event.is_set():
                    break
                self._execute(child, progress=progress, context=context,
                              source_name=source_name, stack=stack)


def creation_dialog(name, store, intent_parser, *, triggers=None,
                    taught_confirmation=False, agent_runner=None):
    """Build the first Question in a side-effect-free voice creation dialog.

    ``intent_parser(answer)`` may return ``(action, arg)`` or a dict with
    ``action``/``do`` and ``arg``. Only workflow-safe primitives are accepted.
    """
    workflow_name = str(name).strip()
    seeded_triggers = []
    for trigger in triggers or [workflow_name]:
        trigger = normalize(trigger)
        if trigger and trigger not in seeded_triggers:
            seeded_triggers.append(trigger)
    steps = []
    failures = 0

    def parse(answer, allow_condition=True):
        heard = normalize(answer)
        for recipe in sorted(store.list(), key=lambda item: len(normalize(item["name"])),
                             reverse=True):
            saved = normalize(recipe["name"])
            if heard in {f"run {saved}", f"run my {saved} workflow",
                         f"run the {saved} workflow", f"run {saved} workflow"}:
                return {"do": "run", "arg": recipe["name"]}
        if allow_condition:
            match = re.match(r"^if (.+?) is open,? (.+)$", str(answer).strip(), re.I)
            if match:
                child = parse(match.group(2), allow_condition=False)
                if child is not None:
                    return {"do": "if_window", "arg": match.group(1).strip(),
                            "then": [child]}
        value = intent_parser(answer)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            action, arg = value[0], value[1]
        elif isinstance(value, dict):
            step = dict(value)
            action = step.pop("action", step.get("do", None))
            step["do"] = action
            arg = step.get("arg")
        else:
            return None
        if action not in PRIMITIVES or action == "ask" or arg is None:
            return None
        if not isinstance(value, dict):
            step = {"do": action, "arg": arg}
        return step if _validate_step(step) else None

    def next_question(step):
        return Question(f"Added {WorkflowRunner.describe(step)}. What is the next step? Say done when finished.", handle)

    def confirm_agent_step(step):
        lane = step.get("lane")
        if lane is None and agent_runner is not None:
            try:
                from agents import select_lane
                lane = select_lane(None, agent_runner.lanes).name
                step["lane"] = lane
            except Exception as exc:
                return str(exc)
        lane = lane or "agent"

        def confirm(answer):
            heard = normalize(answer)
            if heard in {"yes", "yeah", "yep", "sure", "confirm", "keep it"}:
                steps.append(step)
                return next_question(step)
            if heard in {"no", "nope", "cancel", "never mind", "nevermind"}:
                return Question("Okay, what is the next step? Say done when finished.", handle)
            return Question("Please say yes or no.", confirm)
        return Question(f"Step sends a task to {lane}. Keep it?", confirm)

    def handle(answer):
        nonlocal failures
        heard = normalize(answer)
        if heard in {"cancel", "never mind", "nevermind"}:
            return "Workflow creation cancelled."
        if heard == "done":
            if not steps:
                return "Workflow creation cancelled; no steps were added."
            recipe = {"name": workflow_name,
                      "triggers": seeded_triggers, "steps": steps}
            store.save(recipe)
            if taught_confirmation:
                return f"Saved. Say '{seeded_triggers[0]}' any time."
            return f"Saved workflow {workflow_name} with {len(steps)} steps."
        step = parse(answer)
        if step is None:
            failures += 1
            if failures >= 2:
                return "I couldn't understand that step. Workflow creation cancelled."
            return Question("I didn't understand that step. Please try once more, or say done.", handle)
        failures = 0
        if step["do"] == "agent":
            return confirm_agent_step(step)
        steps.append(step)
        return next_question(step)

    return Question(f"Creating workflow {workflow_name}. What is the first step?", handle)


# Friendly aliases for callers that prefer builder terminology.
build_creation_dialog = creation_dialog
