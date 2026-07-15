"""Keep Flow alive. Relaunch flow.py if it exits non-zero (a crash), preserving
the crashed run's full log for diagnosis. Stops on clean exit (tray Quit) or a
crash loop. Runs hidden via pythonw from the Startup shortcut.
"""

import datetime
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# Refuse to run a second supervisor — two of them spawning flow.py was half of
# the "tangled instances" mess. Held for the supervisor's lifetime.
_guard = socket.socket()
try:
    _guard.bind(("127.0.0.1", 18757))
except OSError:
    sys.exit("A Flow supervisor is already running.")

APP = Path(__file__).parent
PYW = str(APP / ".venv" / "Scripts" / "pythonw.exe")
FLOW = str(APP / "flow.py")
LOG = APP / "flow.log"
CRASH = APP / "data" / "flow-crash.log"
CRASH.parent.mkdir(parents=True, exist_ok=True)
CREATE_NO_WINDOW = 0x08000000


def client_self_update():
    """Thin clients pull the latest app files from the Flow server before each
    launch, so improvements to flow.py propagate automatically (like the phone
    APK does). No-op on the host (mode != client)."""
    import json
    import urllib.request
    cfg_path = APP / "config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return
    if cfg.get("mode") != "client":
        return
    url = (cfg.get("server_url") or "").rstrip("/")
    if not url:
        return
    try:
        ver = json.loads(urllib.request.urlopen(
            url + "/client/version", timeout=6).read())["version"]
    except Exception:
        return  # server down / offline — keep running the current files
    vfile = APP / ".client_version"
    if vfile.exists() and vfile.read_text().strip() == ver:
        return
    for f in ("flow.py", "store.py", "llm.py", "commands.py", "supervisor.py",
              "flow.ico"):
        try:
            data = urllib.request.urlopen(url + "/client/file/" + f, timeout=20).read()
            (APP / f).write_bytes(data)
        except Exception:
            return  # partial fetch — don't record the version, retry next launch
    vfile.write_text(ver)
    print(f"[update] client updated to {ver}")


fails = []
while True:
    client_self_update()
    with LOG.open("w", encoding="utf-8") as out:
        code = subprocess.call([PYW, FLOW], cwd=str(APP), stdout=out,
                               stderr=subprocess.STDOUT,
                               creationflags=CREATE_NO_WINDOW)
    if code == 0:
        break  # clean quit (tray) or already-running
    try:
        shutil.copy2(LOG, APP / "data" / "flow-crash-last.log")  # keep the evidence
    except Exception:
        pass
    now = time.time()
    fails = [t for t in fails if now - t < 120] + [now]
    with CRASH.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} flow.py exited {code}; "
                f"restart {len(fails)} in 120s\n")
    if len(fails) >= 5:
        with CRASH.open("a", encoding="utf-8") as f:
            f.write("crash loop — supervisor giving up\n")
        break
    time.sleep(3)
