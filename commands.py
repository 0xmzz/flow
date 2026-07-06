"""Command mode for Flow — route spoken instructions to actions.

This is the seed of "use the computer by voice". Transcribed speech is matched
against skills (crypto price, open, search, ...) that DO something and speak a
result, instead of being pasted as text. Unmatched commands fall through to an
LLM. Each skill is a small, testable unit — add more over time.
"""

import os
import re
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

        # --- open an app or site ---
        m = re.search(r"^(?:open|launch|go to) (.+)", t)
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

    def _open(self, name):
        key = name.replace(" ", "").replace("dot", ".")
        if key in KNOWN_SITES:
            webbrowser.open(KNOWN_SITES[key])
            return f"Opening {name}."
        if "." in key or key.startswith("http"):
            webbrowser.open(key if key.startswith("http") else "https://" + key)
            return f"Opening {name}."
        try:
            os.startfile(name)  # noqa: launches by app name / association
            return f"Opening {name}."
        except Exception:
            return f"I couldn't find {name} to open."

    def _fallback(self, text):
        import requests
        c = self.cfg["cleanup"]
        system = ("You are a terse voice assistant. Answer in one or two spoken "
                  "sentences, no markdown.")
        try:
            if c["provider"] == "ollama":  # fully local on the 5090
                r = requests.post(
                    (c["base_url"] or "http://127.0.0.1:11434") + "/api/chat",
                    json={"model": c["model"], "stream": False, "think": False,
                          "keep_alive": -1,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": text}]},
                    timeout=30,
                )
                return r.json()["message"]["content"].strip()
            key = c.get("api_key") or os.environ.get(c["api_key_env"], "")
            if not key:
                return "I don't have a skill for that yet."
            r = requests.post(
                (c.get("base_url") or "https://api.anthropic.com") + "/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={
                    "model": c["model"],
                    "max_tokens": 200,
                    "system": system,
                    "messages": [{"role": "user", "content": text}],
                    **c.get("extra_body", {}),
                },
                timeout=15,
            )
            return r.json()["content"][0]["text"].strip()
        except Exception:
            return "I couldn't reach the assistant."
