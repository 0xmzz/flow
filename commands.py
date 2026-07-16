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
import time
import urllib.parse
import webbrowser


def speak(text):
    """Speak text via Windows SAPI (hands-free result, no screen needed)."""
    if not text:
        return
    try:
        import comtypes
        import comtypes.client
        comtypes.CoInitialize()
        comtypes.client.CreateObject("SAPI.SpVoice").Speak(text)
    except Exception as e:
        print(f"[speak] {e}")


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


class Router:
    def __init__(self, cfg):
        self.cfg = cfg

    def handle(self, text):
        """Return a short spoken response for a command."""
        t = text.strip().rstrip(".!?").lower()
        # Spoken commands come wrapped in politeness — strip it so the verb
        # patterns see "open whatsapp", not "can you open whatsapp for me".
        t = re.sub(r"^(?:hey |ok |okay |so |um |uh )+", "", t)
        t = re.sub(r"^(?:can|could|would|will) you (?:please )?", "", t)
        t = re.sub(r"^please ", "", t)
        t = re.sub(r"\s+(?:for me|please|now|real quick|quickly)$", "", t)

        # --- crypto price ---
        if any(k in t for k in ("price", "how much", "trading at", "worth")):
            m = (re.search(r"price of (?:the )?(.+)", t)
                 or re.search(r"how much is (?:a |an )?(.+?)(?: worth)?$", t)
                 or re.search(r"what(?:'?s| is) (.+?) (?:currently )?(?:trading )?at", t)
                 or re.search(r"^(.+?) price$", t))
            if m:
                q = m.group(1).strip()
                try:
                    res = _coingecko_price(q)
                except Exception as e:
                    return f"I couldn't reach the price feed: {e}"
                if not res:
                    return f"I couldn't find a price for {q}."
                name, price, chg = res
                chg_s = f", {chg:+.1f} percent in the last day" if chg is not None else ""
                return f"{name} is {_fmt_price(price)}{chg_s}."

        # --- switch to an open window ---
        m = re.search(r"^(?:switch to|go to|focus|bring up) (.+)", t)
        if m:
            title = switch_to(m.group(1).strip())
            if title:
                return f"Switched to {title[:40]}."
            # "go to X" might mean a site — fall through to open
            t2 = m.group(1).strip()
            return self._open(t2)

        # --- open an app or site ---
        m = re.search(r"^(?:open|launch|start) (.+)", t)
        if m:
            return self._open(m.group(1).strip())

        # --- web search ---
        m = re.search(r"^(?:search|google|look up)(?: for)? (.+)", t)
        if m:
            webbrowser.open("https://www.google.com/search?q="
                            + urllib.parse.quote(m.group(1)))
            return f"Searching for {m.group(1)}."

        # --- fallback: ask the LLM ---
        return self._fallback(text)

    FOLDERS = {
        "downloads": "Downloads", "documents": "Documents", "desktop": "Desktop",
        "pictures": "Pictures", "videos": "Videos", "music": "Music",
    }

    def _open(self, name):
        n = name.lower().strip()
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
        key = name.replace(" ", "").replace("dot", ".")
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
            return f"I couldn't find {name} to open."

    def _fallback(self, text):
        import llm
        out, _ = llm.chain(
            self.cfg["cleanup"],
            "You are a terse voice assistant. Answer in one or two spoken "
            "sentences, no markdown.",
            text, max_tokens=200,
            local_only=self.cfg.get("local_only", True))
        return out or "I couldn't reach the assistant."
