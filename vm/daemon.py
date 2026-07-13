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
                        except Exception as e:
                            print(f"start failed: {e}")
                            holding = False
                    elif ev.value == 0 and holding:          # keyup
                        holding = False
                        try:
                            text = post("/record/stop").get("text", "")
                            if text:
                                type_text(text)
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
    main()
