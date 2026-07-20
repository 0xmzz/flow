"""Command mode for Flow — route spoken instructions to actions.

This is the seed of "use the computer by voice". Transcribed speech is matched
against skills (crypto price, open, search, ...) that DO something and speak a
result, instead of being pasted as text. Unmatched commands fall through to an
LLM. Each skill is a small, testable unit — add more over time.
"""

import ctypes
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

from agents import AgentError, AgentRunner, select_lane
from workflows import (Question, RecipeStore, WorkflowRunner, creation_dialog,
                       normalize as normalize_workflow)


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
    "hyperliquid": "https://app.hyperliquid.xyz",
    "google": "https://google.com",
}

ROUTER_VERBS = ("open", "reopen", "launch", "start", "close", "switch", "navigate", "search", "google",
                "price", "type", "write", "help", "run", "put", "move", "place", "arrange")
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
                 intent_parser=None, agent_runner=None, clock=None):
        self.cfg = cfg
        self.workflows = workflow_store or RecipeStore()
        self.workflow_progress_fn = lambda _name, _i, _n, _text: None
        self.workflow_complete_fn = lambda _text: None
        self.workflow_ask_fn = None
        self.intent_parser = intent_parser
        self.agent_progress_fn = lambda _lane, _tail: None
        self.agent_complete_fn = lambda _result: None
        self.agent_runner = agent_runner or AgentRunner()
        self.clock = clock or time.monotonic
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

    def handle(self, text):
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

        if re.search(r"^run .+\bthen\s+(?:ask\s+(?:codex|claude)\s+to|send\s+to\s+(?:codex|claude))\b", t):
            return "I can't split that into commands. Teach it as a workflow instead."

        explicit_workflow = re.match(r"^run\s+(?:my\s+|the\s+)?(.+?)\s+workflow$", t)
        if explicit_workflow:
            return self._run_workflow(explicit_workflow.group(1).strip())

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
            return (f"Running workflow {recipe['name']}." if started
                    else "A workflow is already running.")

        explicit = re.match(
            r"^(?:ask\s+(codex|claude)\s+to|send\s+to\s+(codex|claude))\s+(.+)$",
            text.strip().rstrip(".!?"), re.IGNORECASE)
        if explicit:
            return self._confirm_agent(
                explicit.group(3).strip(),
                (explicit.group(1) or explicit.group(2)).lower())

        # --- crypto price ---
        if any(k in t for k in ("price", "how much", "trading at", "worth")):
            m = (re.search(r"price of (?:the )?(.+)", t)
                 or re.search(r"how much is (?:a |an )?(.+?)(?: worth)?$", t)
                 or re.search(r"what(?:'?s| is) (.+?) (?:currently )?(?:trading )?at", t)
                 or re.search(r"^(.+?) price$", t))
            if m:
                q = _strip_tails(m.group(1).strip())
                return self._price(q)

        # --- discoverability ---
        if re.search(r"^(what can you do|help|commands|what do you do)$", t):
            return ("I can open apps: say open WhatsApp. Switch windows: say "
                    "switch to Chrome. Check crypto: say price of bitcoin. "
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
            return self._switch(m.group(1).strip())

        m = re.search(r"^close (.+)", t)
        if m:
            return self._close(m.group(1).strip())

        # --- open an app or site ---
        m = re.search(r"^(?:open|launch|start) (.+)", t)
        if m:
            names = [re.sub(r"^open\s+", "", part.strip())
                     for part in m.group(1).split(" and ")]
            if len(names) > 1 and all(names):
                return " ".join(filter(None, (self._open(name) for name in names)))
            return self._open(m.group(1).strip())

        # --- web search ---
        m = re.search(r"^(?:search|google|look up)(?: for)? (.+)", t)
        if m:
            return self._search(m.group(1).strip())

        m = re.search(r"^(?:type|write) (.+)", t)
        if m:
            return self._type(m.group(1))

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
        if key in KNOWN_SITES:
            webbrowser.open(KNOWN_SITES[key])
            return f"Opening {name}."
        if "." in key or key.startswith("http"):
            webbrowser.open(key if key.startswith("http") else "https://" + key)
            return f"Opening {name}."
        try:
            os.startfile(name)  # noqa: launches by file association
            return f"Opening {name}."
        except Exception:
            words = set(re.findall(r"\b[\w]+\b", n))
            for site in sorted(KNOWN_SITES, key=len, reverse=True):
                if set(site.split()) <= words:
                    webbrowser.open(KNOWN_SITES[site])
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
        workflow_context = [
            {"name": recipe["name"], "triggers": recipe.get("triggers", [])}
            for recipe in self.workflows.list()
        ]
        raw, _ = llm.chain(
            self.cfg["cleanup"],
            'Extract the voice-command intent. Reply with STRICT JSON only: '
            '{"action":"open|close|switch|price|search|type|keys|workflow|agent|answer",'
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
            'after You said a suggested command, map it to that command action.',
            contextual_text, max_tokens=80,
            local_only=self.cfg.get("local_only", True))
        intent = self._parse_intent(raw)
        if intent and intent[0] != "answer" and intent[1].strip():
            action, arg = intent
            if action == "agent":
                return self._confirm_agent(arg.strip())
            if action == "workflow":
                return self._run_workflow(arg.strip())
            handlers = {"open": self._open, "close": self._close,
                        "switch": self._switch, "price": self._price,
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
                    return self._chat_answer(contextual_text)
                return Question("Please say yes or no.", teach,
                                cancel_text="Okay.")
            return Question("I don't know how to do that yet. Teach me the steps?",
                            teach, cancel_text="Okay.")
        return self._chat_answer(contextual_text)

    def _chat_answer(self, contextual_text):
        import llm
        out, _ = llm.chain(
            self.cfg["cleanup"],
            "You are a terse voice assistant. Answer in one or two spoken sentences, "
            "no markdown. You control this PC ONLY through these listed skills: "
            "open, close, switch, crypto price, web search, type text, browser keys, "
            "workflows, and agent dispatch. Never offer an action outside that list. "
            "Never say you can do an action without doing it. The intent lane already "
            "handles actions; this lane answers questions and gives brief acknowledgements. "
            "If the user seems to want an action, suggest the exact spoken command, for "
            "example: Say: open chrome.",
            contextual_text, max_tokens=200,
            local_only=self.cfg.get("local_only", True))
        return out or "I couldn't reach the assistant."

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
            if (action in {"open", "close", "switch", "price", "search",
                           "type", "keys", "workflow", "agent", "answer"} and isinstance(arg, str)):
                return action, arg
        return None
