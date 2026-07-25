"""Command mode for Flow — route spoken instructions to actions.

This is the seed of "use the computer by voice". Transcribed speech is matched
against skills (crypto price, open, search, ...) that DO something and speak a
result, instead of being pasted as text. Unmatched commands fall through to an
LLM. Each skill is a small, testable unit — add more over time.
"""

import ctypes
import datetime
import difflib
import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from pathlib import Path

try:
    from agents import AgentError, AgentRunner, select_lane
    from workflows import (Question, RecipeStore, WorkflowRunner, creation_dialog,
                           normalize as normalize_workflow)
    HOST_FEATURES = True
except ImportError as _missing:  # pragma: no cover - exercised on thin clients
    # Thin clients ship the dictation core only. Workflows and agent lanes are
    # host features; a client that lacks those modules must still dictate
    # rather than crash-loop on import. (It did: a client pulled a commands.py
    # that hard-imported these and was dead for five days.)
    HOST_FEATURES = False
    print(f"[commands] host features unavailable ({_missing}); dictation only")

    class AgentError(RuntimeError):
        outcome = "error"

    class Question:  # only ever type-checked in this mode, never constructed
        def __init__(self, text, handler=None, cancel=None, cancel_text="Cancelled."):
            self.text, self.handler = text, handler
            self.cancel, self.cancel_text = cancel, cancel_text

    class RecipeStore:
        def names(self):
            return []

        def match(self, _text):
            return None

        def get(self, _name):
            return None

        def delete(self, _name):
            return False

    class _InertRunner:
        running = False
        lanes = ()

        def __init__(self, *_a, **_kw):
            pass

        def start(self, *_a, **_kw):
            return False

        def cancel(self):
            return None

    AgentRunner = WorkflowRunner = _InertRunner

    def select_lane(*_a, **_kw):
        raise AgentError("Agent lanes are not available on this client.")

    def creation_dialog(*_a, **_kw):
        return "Workflows aren't available on this client."

    def normalize_workflow(text):
        cleaned = re.sub(r"[^\w\s]", " ", str(text or "").lower())
        return re.sub(r"\s+", " ", cleaned).strip()


DATA_DIR = Path(__file__).parent / "data"
OFFLINE_LINE = ("My language model is offline — direct commands still work, "
                "say help for the list.")
_intent_log_lock = threading.Lock()
_intent_log_local = threading.local()
_live_state_lock = threading.Lock()


def machine_context(*, workflows=(), workflow_running=False,
                    agent_running=False, now=None, windows_supplier=None,
                    monitors_supplier=None):
    """Return a cheap, fresh, bounded snapshot of useful machine state."""
    now = now or datetime.datetime.now()
    if windows_supplier is None:
        windows_supplier = _windows_list
    if monitors_supplier is None:
        from winctl import enumerate_monitors
        monitors_supplier = enumerate_monitors
    titles = []
    seen = set()
    try:
        for _handle, value in windows_supplier():
            title = " ".join(str(value or "").split())
            key = title.casefold()
            if title and key not in seen:
                seen.add(key)
                titles.append(title[:48])
            if len(titles) == 8:
                break
    except Exception:
        pass
    try:
        monitor_count = len(monitors_supplier())
    except Exception:
        monitor_count = 0
    try:
        names = [" ".join(str(name).split())[:32] for name in workflows if name]
    except Exception:
        names = []
    running = []
    if workflow_running:
        running.append("workflow")
    if agent_running:
        running.append("agent")
    prefix = f"Machine: {now.strftime('%A %H:%M')}; windows: "
    suffix = (f"; monitors: {monitor_count}; "
              f"workflows: {(', '.join(names[:6]) or 'none')[:120]}; "
              f"running: {', '.join(running) or 'none'}")
    window_budget = max(0, 400 - len(prefix) - len(suffix))
    windows = (", ".join(titles) or "none")[:window_budget]
    return (prefix + windows + suffix)[:400]


def _morning_rule(context):
    return (context["first_today"] and 5 <= context["now"].hour < 11
            and context.get("morning_workflow") is not None)


def _morning_question(context):
    return f"Morning. Want me to run {context['morning_workflow']['name']}?"


def _morning_action(router, context):
    return router._run_workflow(context["morning_workflow"]["name"])


GREETING_RULES = [(_morning_rule, _morning_question, _morning_action)]


def speak(text):
    """Speak text synchronously through the configured local TTS lane."""
    from speech import speak as speech_speak
    return speech_speak(text)


def _fmt_price(p):
    if p is None:
        return "an unknown price"
    if p >= 1:
        return f"${p:,.2f}"
    return "$" + f"{p:.8f}".rstrip("0").rstrip(".")


def _coingecko_price(query):
    import requests
    s = requests.get("https://api.coingecko.com/api/v3/search",
                     params={"query": query}, timeout=8).json()
    coins = s.get("coins") or []
    if not coins:
        return None
    coin = coins[0]
    p = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": coin["id"], "vs_currencies": "usd",
                "include_24hr_change": "true"},
        timeout=8,
    ).json()
    d = p.get(coin["id"], {})
    return coin.get("name", coin["id"]), d.get("usd"), d.get("usd_24h_change")


_WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
    96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


def _open_meteo(location):
    """Fetch current and today's weather; only the location string is geocoded."""
    import requests
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=6)
    geo.raise_for_status()
    places = geo.json().get("results") or []
    if not places:
        raise ValueError("location not found")
    place = places[0]
    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": place["latitude"], "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min",
                "forecast_days": 1, "timezone": "auto"}, timeout=6)
    forecast.raise_for_status()
    data = forecast.json()
    current, daily = data["current"], data["daily"]
    return {"place": place.get("name") or location,
            "temp": current["temperature_2m"],
            "feels": current["apparent_temperature"],
            "condition": _WEATHER_CODES.get(current.get("weather_code"),
                                              "unknown conditions"),
            "high": daily["temperature_2m_max"][0],
            "low": daily["temperature_2m_min"][0]}


_app_cache = {"t": 0.0, "apps": {}}


def installed_apps():
    """Start-menu apps: spoken-name -> AppID. Cached 5 min."""
    if time.time() - _app_cache["t"] > 300 or not _app_cache["apps"]:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-StartApps | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=15,
                creationflags=0x08000000).stdout
            data = json.loads(out or "[]")
            if isinstance(data, dict):
                data = [data]
            _app_cache["apps"] = {a["Name"].lower(): a["AppID"]
                                  for a in data if a.get("Name") and a.get("AppID")}
            _app_cache["t"] = time.time()
        except Exception as e:
            print(f"[apps] index failed: {e}")
    return _app_cache["apps"]


def _match(name, candidates):
    n = name.lower().strip()
    if n in candidates:
        return n
    subs = [k for k in candidates if n in k]
    if subs:
        return min(subs, key=len)  # shortest containing match
    m = difflib.get_close_matches(n, list(candidates), n=1, cutoff=0.7)
    return m[0] if m else None


def open_app(name):
    """Launch any installed app by spoken name. Returns its real name or None."""
    apps = installed_apps()
    k = _match(name, apps)
    if not k:
        return None
    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{apps[k]}"],
                     creationflags=0x08000000)
    return k.title()


def _windows_list():
    u = ctypes.windll.user32
    res = []
    proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def cb(h, _l):
        if u.IsWindowVisible(h):
            ln = u.GetWindowTextLengthW(h)
            if ln:
                b = ctypes.create_unicode_buffer(ln + 1)
                u.GetWindowTextW(h, b, ln + 1)
                res.append((h, b.value))
        return True

    u.EnumWindows(proc(cb), 0)
    return res


def switch_to(name):
    """Focus an open window by (fuzzy) title. Returns its title or None."""
    wins = _windows_list()
    titles = {t.lower(): h for h, t in wins if t.strip()}
    k = _match(name, titles)
    if not k:
        return None
    h = titles[k]
    u = ctypes.windll.user32
    u.ShowWindow(h, 9)              # SW_RESTORE
    u.SwitchToThisWindow(h, True)   # reliably steals focus for user-initiated switch
    return k.title()


KNOWN_SITES = {
    "youtube": "https://youtube.com",
    "gmail": "https://mail.google.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "coingecko": "https://coingecko.com",
    "github": "https://github.com",
    "google": "https://google.com",
}

ROUTER_VERBS = ("open", "reopen", "launch", "start", "close", "switch", "navigate", "search", "google",
                "price", "type", "write", "help", "run", "put", "move", "place", "arrange",
                "save", "remember")

APP_EXE_NAMES = {
    "chrome.exe": "chrome", "msedge.exe": "edge",
    "code.exe": "visual studio code", "windowsterminal.exe": "terminal",
    "explorer.exe": "file explorer",
}


def openable_app(exe, title="", apps=None):
    """Map a process/window identity to a name accepted by ``open_app``."""
    exe_key = os.path.basename(str(exe or "")).casefold()
    if exe_key in APP_EXE_NAMES:
        return APP_EXE_NAMES[exe_key]
    apps = installed_apps() if apps is None else apps
    stem = os.path.splitext(exe_key)[0]
    match = _match(stem, apps) if stem else None
    if match:
        return match
    suffix = re.search(r"\s[-–—]\s([^–—-]+)$", str(title or ""))
    match = _match(suffix.group(1).strip(), apps) if suffix else None
    return match


_TAIL_RE = re.compile(
    r"\s+(?:right\s+now|at\s+the\s+moment|real\s+quick|for\s+me|currently|"
    r"quickly|today|please|now)\s*$")


def _strip_tails(text):
    previous = None
    while text != previous:
        previous = text
        text = _TAIL_RE.sub("", text).strip()
    return text


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


def _monitor_number(value):
    value = str(value).lower()
    return _NUMBER_WORDS.get(value, int(value) if value.isdigit() else None)


def _parse_place(text):
    """Parse supported placement phrases into one canonical workflow step."""
    heard = str(text or "").strip().lower().rstrip(".!?")
    region_words = (r"top[ -]left|top[ -]right|bottom[ -]left|bottom[ -]right|"
                    r"left|right|top|bottom|full")
    number = r"\d+|one|two|three|four|five|six|seven|eight|nine"
    patterns = (
        rf"^(?:put|move|place)\s+(.+?)\s+(?:on|in|to)\s+(?:the\s+)?"
        rf"(?:(?:({region_words})(?:\s+half|\s+quarter)?)(?:\s+of)?\s+)?"
        rf"(?:monitor|screen)\s+({number})$",
        rf"^(?:put|move|place)\s+(.+?)\s+(?:on|in|to)\s+(?:the\s+)?"
        rf"(?:monitor|screen)\s+({number})(?:\s*,?\s*(?:on\s+)?(?:the\s+)?"
        rf"({region_words})(?:\s+half|\s+quarter)?)?$",
        rf"^(.+?)\s+({region_words})(?:\s+half|\s+quarter)?(?:\s+(?:on|of)\s+"
        rf"(?:the\s+)?(?:monitor|screen)\s+({number}))?$",
    )
    for index, pattern in enumerate(patterns):
        match = re.match(pattern, heard)
        if not match:
            continue
        groups = match.groups()
        if index == 0:
            name, region, monitor = groups
        elif index == 1:
            name, monitor, region = groups
        else:
            name, region, monitor = groups
        name = re.sub(r"^the\s+", "", name).strip()
        if name and name not in {"monitor", "screen", "desktop"}:
            return {"do": "place", "arg": name,
                    "monitor": _monitor_number(monitor) if monitor else None,
                    "region": (region or "full").replace(" ", "-")}
    return None


def _parse_desktop(text):
    heard = str(text or "").strip().lower().rstrip(".!?")
    match = re.match(r"^(next|previous)\s+(?:virtual\s+)?desktop$", heard)
    if match:
        return match.group(1), None
    match = re.match(r"^(?:go to\s+)?(?:virtual\s+)?desktop\s+"
                     r"(\d+|one|two|three|four|five|six|seven|eight|nine)$", heard)
    return (None, _monitor_number(match.group(1))) if match else None


def _action_shaped(text):
    heard = normalize_workflow(text)
    if not heard or heard.startswith(("can you ", "could you ", "what ", "why ",
                                      "how ", "when ", "where ", "are you ")):
        return False
    verbs = {"open", "put", "move", "place", "set", "setup", "start",
             "arrange", "launch", "close", "switch", "type", "write", "run"}
    return heard.split()[0] in verbs or any(
        re.search(rf"\b{verb}\b", heard) for verb in {"open", "put", "set up", "start", "arrange"})


def _workflow_slug(text):
    words = normalize_workflow(text).split()
    fillers = {"please", "the", "a", "an", "my", "for", "me", "can", "you"}
    useful = [word for word in words if word not in fillers][:4]
    if len(useful) < 2:
        useful = (useful + ["workflow"])[:2]
    return " ".join(useful)


class Router:
    def __init__(self, cfg, *, workflow_store=None, workflow_runner=None,
                 intent_parser=None, agent_runner=None, clock=None,
                 layout_supplier=None, apps_supplier=None, now_fn=None,
                 windows_supplier=None, monitors_supplier=None,
                 state_path=None, intent_log_path=None, export_fn=None):
        self.cfg = cfg
        custom_sites = cfg.get("sites") or {}
        if not isinstance(custom_sites, dict):
            custom_sites = {}
        self.sites = {**KNOWN_SITES, **custom_sites}
        self.workflows = workflow_store or RecipeStore()
        self.workflow_progress_fn = lambda _name, _i, _n, _text: None
        self.workflow_complete_fn = lambda _text: None
        self.workflow_ask_fn = None
        self.intent_parser = intent_parser
        self.agent_progress_fn = lambda _lane, _tail: None
        self.agent_complete_fn = lambda _result: None
        self.agent_runner = agent_runner or AgentRunner()
        self.clock = clock or time.monotonic
        self.layout_supplier = layout_supplier
        self.apps_supplier = apps_supplier or installed_apps
        self.now_fn = now_fn or datetime.datetime.now
        self.windows_supplier = windows_supplier
        self.monitors_supplier = monitors_supplier
        self.state_path = Path(state_path) if state_path else DATA_DIR / "live_state.json"
        self.intent_log_path = (Path(intent_log_path) if intent_log_path else
                                DATA_DIR / "intent_log.jsonl")
        self.export_fn = export_fn
        self._conversation = deque(maxlen=6)
        self._conversation_lock = threading.Lock()
        self._conversation_epoch = 0
        self.workflow_runner = workflow_runner or WorkflowRunner(
            router=self, ask_fn=self._ask_workflow, recipe_store=self.workflows,
            agent_runner=self.agent_runner)
        if getattr(self.workflow_runner, "recipe_store", None) is None:
            self.workflow_runner.recipe_store = self.workflows
        if getattr(self.workflow_runner, "agent_runner", None) is None:
            self.workflow_runner.agent_runner = self.agent_runner

    def reset_conversation(self):
        with self._conversation_lock:
            self._conversation.clear()
            self._conversation_epoch += 1

    def _prune_conversation_locked(self, now):
        while self._conversation and now - self._conversation[0][0] > 120.0:
            self._conversation.popleft()

    def remember_exchange(self, user_text, response_text, epoch=None):
        if not isinstance(response_text, str):
            return
        now = self.clock()
        with self._conversation_lock:
            if epoch is not None and epoch != self._conversation_epoch:
                return
            self._prune_conversation_locked(now)
            self._conversation.append((now, str(user_text), response_text))

    def _conversation_text(self, text, limit=800):
        now = self.clock()
        with self._conversation_lock:
            self._prune_conversation_locked(now)
            exchanges = list(self._conversation)
        current = "User: " + str(text)
        if len(current) >= limit:
            return current[:limit]
        blocks = [f"User: {user}\nYou said: {response}"
                  for _timestamp, user, response in exchanges]
        while blocks and len("\n".join(blocks + [current])) > limit:
            blocks.pop(0)
        combined = "\n".join(blocks + [current])
        return combined[-limit:] if len(combined) > limit else combined

    def handle_live(self, text):
        with self._conversation_lock:
            epoch = self._conversation_epoch
        response = self.handle(text)
        self.remember_exchange(text, response, epoch=epoch)
        return response

    def configure_agents(self, *, progress=None, complete=None):
        if progress is not None:
            self.agent_progress_fn = progress
        if complete is not None:
            self.agent_complete_fn = complete

    def configure_workflows(self, *, progress=None, complete=None, ask=None):
        if progress is not None:
            self.workflow_progress_fn = progress
        if complete is not None:
            self.workflow_complete_fn = complete
        if ask is not None:
            self.workflow_ask_fn = ask

    def _ask_workflow(self, text, cancel_event):
        if self.workflow_ask_fn is None:
            raise RuntimeError("workflow question bridge unavailable")
        return self.workflow_ask_fn(text, cancel_event)

    def _machine_context(self):
        return machine_context(
            workflows=self.workflows.names(),
            workflow_running=self.workflow_runner.running,
            agent_running=self.agent_runner.running,
            now=self.now_fn(), windows_supplier=self.windows_supplier,
            monitors_supplier=self.monitors_supplier)

    def _log_intent(self, text, action, *, arg=None, source, context_chars=None):
        record = {"ts": self.now_fn().isoformat(), "text": str(text),
                  "action": str(action), "source": source}
        if getattr(_intent_log_local, "private", False):
            record["private"] = True
        if arg is not None:
            record["arg"] = str(arg)
        if context_chars is not None:
            record["context_chars"] = int(context_chars)
        try:
            with _intent_log_lock:
                self.intent_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.intent_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[intent-log] failed: {exc}")
        _intent_log_local.count = getattr(_intent_log_local, "count", 0) + 1

    def _fast(self, text, action, response, arg=None):
        self._log_intent(text, action, arg=arg, source="fastpath")
        return response

    def claim_first_conversation_today(self):
        """Atomically record a wake interaction and report if it is today's first."""
        today = self.now_fn().date().isoformat()
        with _live_state_lock:
            state = {}
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
            first = state.get("last_conversation_date") != today
            if first:
                state["last_conversation_date"] = today
                try:
                    self.state_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.state_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(state, indent=2) + "\n",
                                         encoding="utf-8")
                    temporary.replace(self.state_path)
                except OSError as exc:
                    print(f"[live-state] failed: {exc}")
            return first

    def proactive_greeting(self, first_today):
        recipes = self.workflows.list()
        morning = next((recipe for recipe in recipes if any(
            re.search(r"day|morning", str(value), re.IGNORECASE)
            for value in [recipe.get("name", ""), *recipe.get("triggers", [])])), None)
        context = {"now": self.now_fn(), "first_today": bool(first_today),
                   "morning_workflow": morning}
        for predicate, question, action in GREETING_RULES:
            if not predicate(context):
                continue

            def handle(answer, selected=action, values=context):
                heard = normalize_workflow(answer)
                if heard in {"yes", "yeah", "yep", "sure", "okay", "ok"}:
                    return selected(self, values)
                if heard in {"no", "nope", "cancel", "never mind", "nevermind"}:
                    return "Okay."
                return Question("Please say yes or no.", handle)
            return Question(question(context), handle, cancel_text="Okay.")
        return None

    def handle(self, text, *, private=False):
        """Resolve one utterance and ensure the decision corpus gets one row."""
        before = getattr(_intent_log_local, "count", 0)
        old_private = getattr(_intent_log_local, "private", False)
        _intent_log_local.private = bool(private)
        try:
            response = self._handle_impl(text)
            if getattr(_intent_log_local, "count", 0) == before:
                heard = normalize_workflow(text)
                first = heard.split()[0] if heard else "unknown"
                aliases = {"launch": "open", "start": "open", "navigate": "switch",
                           "focus": "switch", "google": "search", "write": "type",
                           "run": "workflow", "save": "save_layout",
                           "remember": "save_layout", "what": "help"}
                self._log_intent(text, aliases.get(first, first), source="fastpath")
            return response
        finally:
            _intent_log_local.private = old_private

    def _handle_impl(self, text):
        """Return a short spoken response for a command."""
        t = text.strip().rstrip(".!?").lower()
        # Spoken commands come wrapped in politeness — strip it so the verb
        # patterns see "open whatsapp", not "can you open whatsapp for me".
        t = re.sub(r"^(?:hey |ok |okay |so |um |uh )+", "", t)
        t = re.sub(r"^(?:can|could|would|will) you (?:please )?", "", t)
        t = re.sub(r"^please ", "", t)
        t = re.sub(r"^(?:(?:actually|also|and|then)\s+)+", "", t)
        t = _strip_tails(t)
        parts = t.split(maxsplit=1)
        if parts and parts[0] not in ROUTER_VERBS:
            heard = parts[0]
            forms = {heard, heard.replace("th", "")}
            if heard.endswith("s"):
                forms.add(heard[:-1])
            scored = [(difflib.SequenceMatcher(None, form, verb).ratio(), verb)
                      for form in forms for verb in ROUTER_VERBS]
            ratio, match = max(scored)
            if ratio >= 0.8:
                t = match + ((" " + parts[1]) if len(parts) == 2 else "")

        if t in {"stop", "cancel"} and self.agent_runner.running:
            self.agent_runner.cancel()
            return "Cancelling agent."
        if t in {"stop", "cancel"} and self.workflow_runner.running:
            self.workflow_runner.cancel()
            return "Cancelling workflow."
        if re.fullmatch(r"(?:flow,?\s+)?export my data", t):
            if self.export_fn is None:
                return self._fast(text, "export", "Data export is not configured.")
            try:
                summary = self.export_fn()
                name = Path(summary["dest"]).name or summary["dest"]
                response = f"Exported {summary['entries']} entries to {name}."
            except Exception as exc:
                response = f"Export failed: {exc}"
            return self._fast(text, "export", response)

        # Workflow management is explicit so recipe triggers cannot shadow it.
        if t == "list workflows":
            names = self.workflows.names()
            return (("Workflows: " + ", ".join(names) +
                     ". Workflows can run other workflows.") if names
                    else "You don't have any workflows yet. Workflows can run other workflows.")
        m = re.search(r"^delete workflow (.+)$", t)
        if m:
            name = m.group(1).strip()
            if not self.workflows.get(name):
                return f"I couldn't find workflow {name}."

            def confirm(answer):
                heard = str(answer).strip().lower().rstrip(".!?")
                if heard in {"yes", "yeah", "yep", "sure", "confirm"}:
                    self.workflows.delete(name)
                    return f"Deleted workflow {name}."
                if heard in {"no", "nope", "cancel", "never mind", "nevermind"}:
                    return "Deletion cancelled."
                return Question("Please say yes or no.", confirm)
            return Question(f"Delete workflow {name}?", confirm)
        m = re.search(r"^create a workflow called (.+)$", t)
        if m:
            return creation_dialog(
                m.group(1).strip(), self.workflows, self.parse_action,
                agent_runner=self.agent_runner)
        m = re.match(r"^(?:save this (?:layout|setup)|remember this layout) as (.+)$", t)
        if m:
            return self._save_layout(m.group(1).strip())

        if re.search(r"^run .+\bthen\s+(?:ask\s+(?:codex|claude)\s+to|send\s+to\s+(?:codex|claude))\b", t):
            return "I can't split that into commands. Teach it as a workflow instead."

        explicit_workflow = re.match(r"^run\s+(?:my\s+|the\s+)?(.+?)\s+workflow$", t)
        if explicit_workflow:
            arg = explicit_workflow.group(1).strip()
            return self._fast(text, "workflow", self._run_workflow(arg), arg)

        desktop = _parse_desktop(t)
        if desktop:
            direction, requested = desktop
            if requested is not None:
                return (f"Windows only exposes next and previous desktop by key; "
                        f"I can't reliably jump to desktop {requested}.")
            return self._desktop(direction)

        placement = _parse_place(t)
        if placement:
            return self._place_step(placement)

        recipe = self.workflows.match(t)
        if recipe:
            if self.workflow_runner.running:
                return "A workflow is already running."
            started = self.workflow_runner.start(
                recipe, progress=lambda i, total, description:
                self.workflow_progress_fn(recipe["name"], i, total, description),
                complete=self.workflow_complete_fn)
            response = (f"Running workflow {recipe['name']}." if started
                        else "A workflow is already running.")
            return self._fast(text, "workflow", response, recipe["name"])

        explicit = re.match(
            r"^(?:ask\s+(codex|claude)\s+to|send\s+to\s+(codex|claude))\s+(.+)$",
            text.strip().rstrip(".!?"), re.IGNORECASE)
        if explicit:
            return self._fast(text, "agent", self._confirm_agent(
                explicit.group(3).strip(),
                (explicit.group(1) or explicit.group(2)).lower()),
                explicit.group(3).strip())

        # --- weather ---
        weather = (re.match(r"^weather in (.+)$", t)
                   or re.match(r"^(?:what(?:'?s| is) )?(?:the )?weather(?: today| now| like)?$", t))
        if weather:
            place = (weather.group(1).strip() if weather.lastindex else
                     self.cfg.get("weather_location", "Dubai"))
            return self._fast(text, "weather", self._weather(place), place)

        # --- crypto price ---
        if any(k in t for k in ("price", "how much", "trading at", "worth")):
            m = (re.search(r"price of (?:the )?(.+)", t)
                 or re.search(r"how much is (?:a |an )?(.+?)(?: worth)?$", t)
                 or re.search(r"what(?:'?s| is) (.+?) (?:currently )?(?:trading )?at", t)
                 or re.search(r"^(.+?) price$", t))
            if m:
                q = _strip_tails(m.group(1).strip())
                return self._fast(text, "price", self._price(q), q)

        # --- discoverability ---
        if re.search(r"^(what can you do|help|commands|what do you do)$", t):
            return ("I can open apps: say open WhatsApp. Switch windows: say "
                    "switch to Chrome. Check crypto: say price of bitcoin. "
                    "Check weather: say weather in Paris. "
                    "Search the web: say search for something. Anything else, "
                    "I answer with the local model.")

        key_commands = {
            "new tab": "ctrl+t", "open a new tab": "ctrl+t",
            "open new tab": "ctrl+t", "reopen tab": "ctrl+shift+t",
            "next tab": "ctrl+tab", "previous tab": "ctrl+shift+tab",
            "refresh": "f5", "reload": "f5", "go back": "alt+left",
            "go forward": "alt+right",
        }
        if t in key_commands:
            return self._keys(key_commands[t])

        # --- switch to an open window ---
        m = re.search(r"^(?:switch to|go to|navigate to|focus|bring up) (.+)", t)
        if m:
            arg = m.group(1).strip()
            return self._fast(text, "switch", self._switch(arg), arg)

        m = re.search(r"^close (.+)", t)
        if m:
            arg = m.group(1).strip()
            return self._fast(text, "close", self._close(arg), arg)

        # --- open an app or site ---
        m = re.search(r"^(?:open|launch|start) (.+)", t)
        if m:
            names = [re.sub(r"^open\s+", "", part.strip())
                     for part in m.group(1).split(" and ")]
            if len(names) > 1 and all(names):
                response = " ".join(filter(None, (self._open(name) for name in names)))
                return self._fast(text, "open", response, ", ".join(names))
            arg = m.group(1).strip()
            return self._fast(text, "open", self._open(arg), arg)

        # --- web search ---
        m = re.search(r"^(?:search|google|look up)(?: for)? (.+)", t)
        if m:
            arg = m.group(1).strip()
            return self._fast(text, "search", self._search(arg), arg)

        m = re.search(r"^(?:type|write) (.+)", t)
        if m:
            arg = m.group(1)
            return self._fast(text, "type", self._type(arg), arg)

        # --- fallback: ask the LLM ---
        return self._fallback(text)

    def parse_action(self, text):
        """Resolve a spoken creation step without executing it."""
        if self.intent_parser is not None:
            return self.intent_parser(text)
        raw = str(text or "").strip()
        normalized = raw.rstrip(".!?")
        explicit_agent = re.match(
            r"^(?:ask\s+(codex|claude)\s+to|send\s+to\s+(codex|claude))\s+(.+)$",
            normalized, re.IGNORECASE)
        if explicit_agent:
            return {"do": "agent", "arg": explicit_agent.group(3).strip(),
                    "lane": (explicit_agent.group(1) or explicit_agent.group(2)).lower()}
        placement = _parse_place(normalized)
        if placement:
            return placement
        desktop = _parse_desktop(normalized)
        if desktop and desktop[0]:
            return {"do": "desktop", "arg": desktop[0]}
        wait = re.match(r"^wait\s+(\d+(?:\.\d+)?|one|two|three|four|five|six|"
                        r"seven|eight|nine)\s*(?:seconds?|secs?)?$",
                        normalized, re.IGNORECASE)
        if wait:
            value = wait.group(1).lower()
            seconds = _monitor_number(value) if not value.replace(".", "").isdigit() else float(value)
            return {"do": "wait_s", "arg": seconds}
        patterns = (
            ("switch", r"^(?:switch to|go to|navigate to|focus|bring up) (.+)"),
            ("close", r"^close (.+)"),
            ("open", r"^(?:open|launch|start) (.+)"),
            ("search", r"^(?:search|google|look up)(?: for)? (.+)"),
            ("type", r"^(?:type|write) (.+)"),
        )
        for action, pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return action, match.group(1).strip()
        if re.search(r"price|how much|trading at|worth", normalized,
                     re.IGNORECASE):
            match = (re.search(r"price of (?:the )?(.+)", normalized, re.I)
                     or re.search(r"^(.+?) price$", normalized, re.I))
            if match:
                return "price", match.group(1).strip()
        import llm
        value, _ = llm.chain(
            self.cfg["cleanup"],
            'Extract the action only. Reply with STRICT JSON: '
            '{"action":"open|close|switch|price|search|type|answer",'
            '"arg":"<string>"}.', raw, max_tokens=80,
            local_only=self.cfg.get("local_only", True))
        intent = self._parse_intent(value)
        return intent if intent and intent[0] != "answer" else None

    FOLDERS = {
        "downloads": "Downloads", "documents": "Documents", "desktop": "Desktop",
        "pictures": "Pictures", "videos": "Videos", "music": "Music",
    }

    def _price(self, query):
        q = _strip_tails(query.strip())
        try:
            res = _coingecko_price(q)
        except Exception as e:
            return f"I couldn't reach the price feed: {e}"
        if not res:
            return f"I couldn't find a price for {q}."
        name, price, chg = res
        chg_s = f", {chg:+.1f} percent in the last day" if chg is not None else ""
        return f"{name} is {_fmt_price(price)}{chg_s}."

    def _weather(self, location):
        try:
            value = _open_meteo(location)
            return (f"{value['place']} is {value['temp']:.0f} degrees, feels like "
                    f"{value['feels']:.0f}, {value['condition']}. Today's high is "
                    f"{value['high']:.0f} and low is {value['low']:.0f}.")
        except Exception:
            return "The weather service didn't answer."

    def _switch(self, name):
        title = switch_to(name)
        if title:
            return f"Switched to {title[:40]}."
        return self._open(name)

    def _search(self, query):
        webbrowser.open("https://www.google.com/search?q="
                        + urllib.parse.quote(query))
        return f"Searching for {query}."

    def _close(self, name):
        n = re.sub(r"^the\s+", "", name.lower().strip())
        if n in ("tab", "window"):
            import keyboard
            keyboard.send("ctrl+w" if n == "tab" else "alt+f4")
            return ""
        wins = _windows_list()
        titles = {title.lower(): hwnd for hwnd, title in wins if title.strip()}
        key = _match(n, titles)
        if not key:
            return f"I couldn't find {name} to close."
        ctypes.windll.user32.PostMessageW(titles[key], 0x0010, 0, 0)
        return f"Closing {key.title()[:40]}."

    def _type(self, text):
        import keyboard
        import pyperclip
        try:
            old = pyperclip.paste()
        except Exception:
            old = None
        try:
            pyperclip.copy(text)
            time.sleep(0.02)
            keyboard.send("ctrl+v")
        finally:
            if old is not None:
                time.sleep(0.05)
                pyperclip.copy(old)
        return ""

    def _place_step(self, step):
        from winctl import place_window
        try:
            placed = place_window(step["arg"], monitor=step.get("monitor"),
                                  region=step.get("region", "full"))
        except (OSError, RuntimeError, ValueError) as exc:
            return f"I couldn't place {step['arg']}: {exc}"
        if not placed:
            return f"I couldn't find {step['arg']} to place."
        region = placed["region"].replace("-", " ")
        region_text = "full screen" if region == "full" else (
            region if " " in region else f"{region} half")
        return (f"{placed['title'][:40]} on monitor {placed['monitor']}, "
                f"{region_text}.")

    def _place_rect_step(self, step):
        from winctl import place_window_rect
        try:
            placed = place_window_rect(
                step["arg"], step["monitor"], step["x"], step["y"],
                step["w"], step["h"])
        except (OSError, RuntimeError, ValueError) as exc:
            return f"I couldn't place {step['arg']}: {exc}"
        return (f"{placed['title'][:40]} on monitor {placed['monitor']}, custom position."
                if placed else f"I couldn't find {step['arg']} to place.")

    def _save_layout(self, name):
        from winctl import snapshot_layout
        snapshot = (self.layout_supplier or snapshot_layout)()
        apps = self.apps_supplier()
        grouped = {}
        for item in snapshot:
            grouped.setdefault(str(item.get("exe") or "").casefold(), []).append(item)
        steps, summary, skipped = [], [], []
        for items in grouped.values():
            first = items[0]
            app = openable_app(first["exe"], first["title"], apps)
            if not app:
                skipped.extend(item["title"] for item in items)
                continue
            titles = {normalize_workflow(item["title"]) for item in items}
            query = first["title"] if len(titles) > 1 else app
            steps.extend([{"do": "open", "arg": app},
                          {"do": "wait_window", "arg": query, "timeout_s": 15}])
            if first.get("do") == "place_rect":
                steps.append({key: first[key] for key in
                              ("do", "monitor", "x", "y", "w", "h")} |
                             {"arg": query})
                where = "custom position"
            else:
                steps.append({"do": "place", "arg": query,
                              "monitor": first["monitor"], "region": first["region"]})
                where = first["region"].replace("-", " ")
            summary.append(f"{app} monitor {first['monitor']} {where}")
        if not steps:
            detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
            return "I couldn't find any openable windows to save." + detail
        recipe = {"name": name, "triggers": [name], "steps": steps,
                  "comment": "Captured from the current virtual desktop only."}

        def save():
            self.workflows.save(recipe)
            result = f"Saved {len(summary)} windows: {', '.join(summary)}. Say '{name}' any time."
            if skipped:
                result += " Skipped: " + ", ".join(skipped) + "."
            return result

        if self.workflows.get(name):
            def replace(answer):
                heard = normalize_workflow(answer)
                if heard in {"replace", "yes", "confirm"}:
                    return save()
                if heard in {"cancel", "no", "never mind", "nevermind"}:
                    return "Layout save cancelled."
                return Question("Please say replace or cancel.", replace)
            return Question(f"A workflow named {name} already exists. Replace or cancel?",
                            replace)
        return save()

    @staticmethod
    def _desktop(direction):
        if direction not in {"next", "previous"}:
            raise ValueError("desktop direction must be next or previous")
        import keyboard
        keyboard.send("win+ctrl+right" if direction == "next" else
                      "win+ctrl+left")
        return f"{direction.title()} desktop."

    @staticmethod
    def _keys(keys):
        import keyboard
        keyboard.send(keys)
        return ""

    def _open(self, name):
        name = re.sub(r"\s+(?:in the browser|in chrome|on the web|online)\s*$",
                      "", name, flags=re.IGNORECASE).strip()
        n = name.lower()
        # 0) drives and common folders ("open c drive", "open downloads")
        m = re.match(r"^(?:the )?([a-z])(?: |-)?drive$", n)
        if m:
            os.startfile(f"{m.group(1).upper()}:\\")
            return f"Opening {m.group(1).upper()} drive."
        if n in self.FOLDERS:
            os.startfile(os.path.join(os.path.expanduser("~"), self.FOLDERS[n]))
            return f"Opening {n}."
        if n in ("file explorer", "explorer", "my computer", "this pc"):
            os.startfile("explorer")
            return "Opening File Explorer."
        # 1) an already-open window? switch instead of spawning a second copy
        title = switch_to(name)
        if title:
            return f"Switched to {title[:40]}."
        # 2) an installed app (anything in the Start menu, fuzzy-matched)
        app = open_app(name)
        if app:
            return f"Opening {app}."
        # 3) known sites / domains
        key = n.replace(" ", "").replace("dot", ".")
        if key in self.sites:
            webbrowser.open(self.sites[key])
            return f"Opening {name}."
        if "." in key or key.startswith("http"):
            webbrowser.open(key if key.startswith("http") else "https://" + key)
            return f"Opening {name}."
        try:
            os.startfile(name)  # noqa: launches by file association
            return f"Opening {name}."
        except Exception:
            words = set(re.findall(r"\b[\w]+\b", n))
            for site in sorted(self.sites, key=len, reverse=True):
                if set(site.split()) <= words:
                    webbrowser.open(self.sites[site])
                    return f"Opening {site}."
            apps = installed_apps()
            for app in sorted(apps, key=len, reverse=True):
                if re.search(rf"\b{re.escape(app)}\b", n):
                    opened = open_app(app)
                    if opened:
                        return f"Opening {opened}."
            return f"I couldn't find {name} to open."

    def _run_workflow(self, name):
        recipe = self.workflows.get(name) or self.workflows.match(name)
        if not recipe:
            return f"I couldn't find workflow {name}."
        if self.workflow_runner.running:
            return "A workflow is already running."
        started = self.workflow_runner.start(
            recipe, progress=lambda i, total, description:
            self.workflow_progress_fn(recipe["name"], i, total, description),
            complete=self.workflow_complete_fn)
        return (f"Running workflow {recipe['name']}." if started
                else "A workflow is already running.")

    def _fallback(self, text):
        import llm
        contextual_text = self._conversation_text(text)
        state = self._machine_context()
        workflow_context = [
            {"name": recipe["name"], "triggers": recipe.get("triggers", [])}
            for recipe in self.workflows.list()
        ]
        raw, lane = llm.chain(
            self.cfg["cleanup"],
            state + '\nExtract the voice-command intent. Reply with STRICT JSON only: '
            '{"action":"open|close|switch|price|weather|search|type|keys|workflow|save_layout|agent|answer",'
            '"arg":"<string>"}. Available workflows: '
            + json.dumps(workflow_context) + '. Use workflow for requests '
            'to run a named workflow or recipe. Never split a request that runs '
            'a workflow and then asks an agent; suggest teaching one workflow. '
            'Agent is ONLY for a concrete, '
            'delegable task with a deliverable. Capability or meta questions '
            '(can you make workflows, are you able to X, what can you do) are '
            'always answer, never agent. Use answer for all questions and conversation. '
            'Keys arg must be one of: new tab, reopen tab, next tab, previous tab, '
            'refresh, go back, go forward. If the current utterance is an affirmation '
            'after You said a suggested command, map it to that command action. '
            'Examples: "price of bitcoin" -> price; "what is the weather like" '
            '-> weather; "weather in Paris" -> weather with arg Paris. '
            'Use the machine snapshot to ground answers such as which tabs or windows '
            'are open and prefer workflow/skill actions using real window names.',
            contextual_text, max_tokens=80,
            cloud_system='Extract intent as strict JSON with action and arg; no machine context.',
            local_only=self.cfg.get("local_only", True))
        if raw is None and lane is None:
            self._log_intent(text, "offline", arg="", source="llm",
                             context_chars=len(state))
            return OFFLINE_LINE
        intent = self._parse_intent(raw)
        if intent and intent[0] != "answer" and intent[1].strip():
            action, arg = intent
            self._log_intent(text, action, arg=arg, source="llm",
                             context_chars=len(state))
            if action == "agent":
                return self._confirm_agent(arg.strip())
            if action == "workflow":
                return self._run_workflow(arg.strip())
            if action == "save_layout":
                return self._save_layout(arg.strip())
            handlers = {"open": self._open, "close": self._close,
                        "switch": self._switch, "price": self._price,
                        "weather": self._weather,
                        "search": self._search, "type": self._type,
                        "keys": self._named_keys}
            return handlers[action](arg.strip())
        if intent and intent[0] == "answer" and _action_shaped(text):
            trigger = normalize_workflow(text)
            name = _workflow_slug(text)

            def teach(answer):
                heard = normalize_workflow(answer)
                if heard in {"yes", "yeah", "yep", "sure", "okay", "ok"}:
                    return creation_dialog(
                        name, self.workflows, self.parse_action,
                        triggers=[trigger], taught_confirmation=True,
                        agent_runner=self.agent_runner)
                if heard in {"no", "nope", "cancel", "never mind", "nevermind"}:
                    return self._chat_answer(contextual_text, state)[0]
                return Question("Please say yes or no.", teach,
                                cancel_text="Okay.")
            self._log_intent(text, "answer", arg="", source="llm",
                             context_chars=len(state))
            return Question("I don't know how to do that yet. Teach me the steps?",
                            teach, cancel_text="Okay.")
        answer, _lane = self._chat_answer(contextual_text, state)
        self._log_intent(text, "answer" if answer != OFFLINE_LINE else "offline",
                         arg="", source="llm", context_chars=len(state))
        return answer

    def _chat_answer(self, contextual_text, state):
        import llm
        out, _ = llm.chain(
            self.cfg["cleanup"],
            state + "\nYou are a terse voice assistant. Answer in one or two spoken sentences, "
            "no markdown. You control this PC ONLY through these listed skills: "
            "open, close, switch, crypto price, web search, type text, browser keys, "
            "workflows, and agent dispatch. Never offer an action outside that list. "
            "Never say you can do an action without doing it. The intent lane already "
            "handles actions; this lane answers questions and gives brief acknowledgements. "
            "If the user seems to want an action, suggest the exact spoken command, for "
            "example: Say: open chrome. Use the machine snapshot to ground answers "
            "such as which tabs or windows are open and prefer workflow/skill actions "
            "using real window names.",
            contextual_text, max_tokens=200,
            cloud_system="You are a terse voice assistant. Answer in one or two spoken sentences.",
            local_only=self.cfg.get("local_only", True))
        return out or OFFLINE_LINE, _

    def _named_keys(self, operation):
        mapping = {
            "new tab": "ctrl+t", "reopen tab": "ctrl+shift+t",
            "next tab": "ctrl+tab", "previous tab": "ctrl+shift+tab",
            "refresh": "f5", "reload": "f5", "go back": "alt+left",
            "go forward": "alt+right",
        }
        key = operation.lower().strip()
        return self._keys(mapping[key]) if key in mapping else ""

    def _confirm_agent(self, task, lane=None):
        if self.agent_runner.running:
            return "An agent is already running."
        try:
            selected = select_lane(lane, self.agent_runner.lanes)
        except AgentError as exc:
            return str(exc)
        preview = " ".join(task.split())
        preview = preview[:57].rstrip() + ("..." if len(preview) > 60 else "")

        def confirm(answer):
            heard = str(answer).strip().lower().rstrip(".!?")
            if heard in {"yes", "yeah", "yep", "sure", "confirm"}:
                try:
                    started = self.agent_runner.start(
                        task, selected.name, progress=self.agent_progress_fn,
                        complete=self.agent_complete_fn)
                except AgentError as exc:
                    return str(exc)
                return (f"Sending to {selected.name}." if started else
                        "An agent is already running.")
            if heard in {"no", "nope", "cancel", "never mind", "nevermind"}:
                return "Okay, not sending."
            return Question("Please say yes or no.", confirm,
                            cancel_text="Okay, not sending.")
        return Question(f"Send to {selected.name}: {preview}?", confirm,
                        cancel_text="Okay, not sending.")

    @staticmethod
    def _parse_intent(raw):
        if not isinstance(raw, str):
            return None
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(value, dict):
                continue
            action, arg = value.get("action"), value.get("arg")
            if (action in {"open", "close", "switch", "price", "weather", "search",
                           "type", "keys", "workflow", "save_layout", "agent", "answer"} and isinstance(arg, str)):
                return action, arg
        return None
