"""LLM provider chain for Flow.

Primary is the local model on the 5090 (Ollama); an optional `fallback` block
in the same shape takes over when the local call fails (Ollama stopped, model
evicted, timeout). Each block: provider ollama|anthropic|openai, base_url,
model, api_key / api_key_env, extra_body.
"""

import os

import requests


def call_provider(p, system, text, max_tokens=2048):
    """One LLM call against a provider block. Raises on any failure."""
    provider = p.get("provider", "ollama")
    if provider == "ollama":
        r = requests.post(
            (p.get("base_url") or "http://127.0.0.1:11434") + "/api/chat",
            json={"model": p["model"], "stream": False, "think": False,
                  "keep_alive": -1,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": text}],
                  **p.get("extra_body", {})},
            timeout=20,
        )
        r.raise_for_status()
        out = r.json()["message"]["content"].strip()
    elif provider == "anthropic":
        key = p.get("api_key") or os.environ.get(p.get("api_key_env", ""), "")
        r = requests.post(
            (p.get("base_url") or "https://api.anthropic.com") + "/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={"model": p["model"], "max_tokens": max_tokens,
                  "system": system,
                  "messages": [{"role": "user", "content": text}],
                  **p.get("extra_body", {})},
            timeout=20,
        )
        r.raise_for_status()
        out = r.json()["content"][0]["text"].strip()
    else:  # openai-compatible
        key = p.get("api_key") or os.environ.get(p.get("api_key_env", ""), "")
        r = requests.post(
            (p.get("base_url") or "https://api.openai.com/v1") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": p["model"],
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": text}],
                  **p.get("extra_body", {})},
            timeout=20,
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
    if not out:
        raise ValueError("empty response")
    return out


def chain(cleanup_cfg, system, text, max_tokens=2048, cloud_system=None):
    """Try primary (local) then fallback. Returns (text, lane) or (None, None).

    cloud_system: if given, the NON-local fallback lane uses this system prompt
    instead of `system`. Callers pass a sanitized prompt here so app/window/
    screen context injected for the local model never leaves the box on the
    cloud fallback path. (The local lane always gets the full `system`.)
    """
    local_is_cloud = cleanup_cfg.get("provider") != "ollama"
    lanes = [("local", cleanup_cfg)]
    fb = cleanup_cfg.get("fallback")
    if fb and fb.get("model"):
        lanes.append(("fallback", fb))
    for name, p in lanes:
        leaves_box = name != "local" or local_is_cloud
        sys_prompt = (cloud_system if (leaves_box and cloud_system is not None)
                      else system)
        try:
            return call_provider(p, sys_prompt, text, max_tokens), \
                f"{name}:{p.get('model')}"
        except Exception as e:
            print(f"[llm] {name} ({p.get('model')}) failed: {e}")
    return None, None
