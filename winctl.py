"""Small, dependency-free helpers for Windows monitor and window placement."""

import ctypes


REGIONS = frozenset({
    "full", "left", "right", "top", "bottom",
    "top-left", "top-right", "bottom-left", "bottom-right",
})


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", RECT),
                ("rcWork", RECT), ("dwFlags", ctypes.c_ulong)]


def region_rect(work_area, region="full"):
    """Return ``(x, y, width, height)`` for a named work-area region."""
    left, top, right, bottom = map(int, work_area)
    region = str(region or "full").lower().replace("_", "-").replace(" ", "-")
    if region not in REGIONS:
        raise ValueError(f"unknown window region: {region}")
    mid_x = left + (right - left) // 2
    mid_y = top + (bottom - top) // 2
    boxes = {
        "full": (left, top, right, bottom),
        "left": (left, top, mid_x, bottom),
        "right": (mid_x, top, right, bottom),
        "top": (left, top, right, mid_y),
        "bottom": (left, mid_y, right, bottom),
        "top-left": (left, top, mid_x, mid_y),
        "top-right": (mid_x, top, right, mid_y),
        "bottom-left": (left, mid_y, mid_x, bottom),
        "bottom-right": (mid_x, mid_y, right, bottom),
    }
    x1, y1, x2, y2 = boxes[region]
    return x1, y1, x2 - x1, y2 - y1


def enumerate_monitors(user32=None):
    """Enumerate monitor work areas in the order supplied by Windows."""
    user32 = user32 or ctypes.windll.user32
    monitors = []
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(RECT), ctypes.c_void_p)

    def visit(handle, _hdc, _rect, _data):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            work = info.rcWork
            monitors.append((len(monitors) + 1, work.left, work.top,
                             work.right, work.bottom, bool(info.dwFlags & 1)))
        return True

    callback = callback_type(visit)
    user32.EnumDisplayMonitors(0, 0, callback, 0)
    return monitors


def place_window(title_query, monitor=None, region="full", *, user32=None,
                 windows_supplier=None, matcher=None, monitors=None):
    """Fuzzy-match and move an open window; return placement metadata or None."""
    if windows_supplier is None or matcher is None:
        from commands import _match, _windows_list
        windows_supplier = windows_supplier or _windows_list
        matcher = matcher or _match
    user32 = user32 or ctypes.windll.user32
    windows = windows_supplier()
    titles = {title.lower(): (handle, title) for handle, title in windows if title.strip()}
    key = matcher(title_query, titles)
    if not key:
        return None
    available = list(monitors) if monitors is not None else enumerate_monitors(user32)
    if not available:
        raise RuntimeError("Windows reported no monitors")
    target_index = int(monitor) if monitor is not None else next(
        (item[0] for item in available if item[5]), available[0][0])
    target = next((item for item in available if item[0] == target_index), None)
    if target is None:
        raise ValueError(f"monitor {target_index} is not available")
    handle, title = titles[key]
    if user32.IsIconic(handle):
        user32.ShowWindow(handle, 9)  # SW_RESTORE
    x, y, width, height = region_rect(target[1:5], region)
    if not user32.MoveWindow(handle, x, y, width, height, True):
        raise OSError("MoveWindow failed")
    return {"title": title, "monitor": target_index,
            "region": str(region).lower().replace(" ", "-")}
