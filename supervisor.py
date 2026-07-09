"""Keep Flow alive. Relaunch flow.py if it exits non-zero (a crash), preserving
the crashed run's full log for diagnosis. Stops on clean exit (tray Quit) or a
crash loop. Runs hidden via pythonw from the Startup shortcut.
"""

import datetime
import shutil
import subprocess
import time
from pathlib import Path

APP = Path(__file__).parent
PYW = str(APP / ".venv" / "Scripts" / "pythonw.exe")
FLOW = str(APP / "flow.py")
LOG = APP / "flow.log"
CRASH = APP / "data" / "flow-crash.log"
CRASH.parent.mkdir(parents=True, exist_ok=True)
CREATE_NO_WINDOW = 0x08000000

fails = []
while True:
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
