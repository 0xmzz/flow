#!/usr/bin/env python3
"""vmflow — Flow bridge daemon for a Linux VM.

Hold RIGHT CTRL inside the VM: the host PC (which owns the mic + GPU) records
and transcribes; the text is typed into the focused VM window via ydotool.
Runs as a root systemd service (evdev + uinput need it).

Env (/etc/vmflow.env): FLOW_HOST (url or "auto" = default gateway), FLOW_TOKEN,
YDOTOOL_SOCKET.
"""

import json
import os
import selectors
import subprocess
import time
import urllib.request

from evdev import InputDevice, ecodes, list_devices

TOKEN = os.environ.get("FLOW_TOKEN", "")
YD = os.environ.get("YDOTOOL_SOCKET", "/run/ydotoold.sock")


def host_url():
    v = os.environ.get("FLOW_HOST", "auto")
    if v != "auto":
        return v.rstrip("/")
    # The Hyper-V host is always the guest's default gateway, whatever subnet
    # the Default Switch moved to after the last host reboot.
    out = subprocess.run(["ip", "route"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("default"):
            return "http://" + line.split()[2] + ":8756"
    return "http://172.24.16.1:8756"


def post(path, timeout=90):
    req = urllib.request.Request(
        host_url() + path, data=b"", method="POST",
        headers={"Authorization": "Bearer " + TOKEN, "X-Device": "work-vm"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path, timeout=6):
    req = urllib.request.Request(
        host_url() + path,
        headers={"Authorization": "Bearer " + TOKEN, "X-Device": "work-vm"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def self_update():
    """Pull the latest daemon from the host; restart into it if it changed.
    Guarded by an env flag so a bad fetch can't restart-loop."""
    import os as _os
    if _os.environ.get("VMFLOW_UPDATED") == "1":
        return
    try:
        req = urllib.request.Request(host_url() + "/vm/daemon.py")
        new = urllib.request.urlopen(req, timeout=6).read()
        cur = open(__file__, "rb").read()
        if new and new.replace(b"\r\n", b"\n") != cur.replace(b"\r\n", b"\n"):
            open(__file__, "wb").write(new)
            print("vmflow: updated, restarting")
            _os.environ["VMFLOW_UPDATED"] = "1"
            _os.execv(sys.executable, [sys.executable, __file__])
    except Exception as e:
        print(f"vmflow: self-update skipped ({e})")


class LiveTyper:
    """While the key is held, poll /record/partial and type committed deltas —
    committed text never retracts, so typing it early is safe."""

    def __init__(self):
        self.typed = 0
        self._active = False
        self._t = None

    def start(self):
        import threading
        self.typed = 0
        self._active = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._active:
            time.sleep(0.5)
            try:
                committed = get("/record/partial").get("committed", "")
            except Exception:
                continue
            delta = committed[self.typed:]
            if delta and self._active:
                type_text(delta)
                self.typed = len(committed)

    def stop(self):
        self._active = False
        if self._t:
            self._t.join(1.5)
        return self.typed


def type_text(text):
    env = dict(os.environ, YDOTOOL_SOCKET=YD)
    subprocess.run(["ydotool", "type", "--", text], env=env, timeout=20)


def keyboards():
    devs = []
    for p in list_devices():
        try:
            d = InputDevice(p)
            if ecodes.KEY_RIGHTCTRL in d.capabilities().get(ecodes.EV_KEY, []):
                devs.append(d)
        except Exception:
            pass
    return devs


def main():
    while True:
        devs = keyboards()
        if devs:
            break
        print("vmflow: no keyboard devices yet, retrying...")
        time.sleep(3)
    sel = selectors.DefaultSelector()
    for d in devs:
        sel.register(d, selectors.EVENT_READ)
    print(f"vmflow: {len(devs)} keyboard(s), host={host_url()}")
    holding = False
    typer = LiveTyper()
    while True:
        for key, _ in sel.select():
            dev = key.fileobj
            try:
                for ev in dev.read():
                    if ev.type != ecodes.EV_KEY or ev.code != ecodes.KEY_RIGHTCTRL:
                        continue
                    if ev.value == 1 and not holding:       # keydown
                        holding = True
                        try:
                            post("/record/start", timeout=10)
                            typer.start()   # live-type committed words as you speak
                        except Exception as e:
                            print(f"start failed: {e}")
                            holding = False
                    elif ev.value == 0 and holding:          # keyup
                        holding = False
                        typed = typer.stop()
                        try:
                            text = post("/record/stop").get("text", "")
                            remainder = text[typed:] if text else ""
                            if remainder:
                                type_text(remainder)
                        except Exception as e:
                            print(f"stop failed: {e}")
                            try:
                                post("/record/cancel", timeout=5)
                            except Exception:
                                pass
                    # ev.value == 2 is auto-repeat: ignored
            except OSError:
                print("keyboard read error; rescanning devices")
                time.sleep(1)
                return main()


if __name__ == "__main__":
    self_update()
    main()
