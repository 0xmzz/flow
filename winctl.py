"""Small, dependency-free helpers for Windows monitor and window placement."""

import ctypes
import os


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


def _iou(a, b):
    """Intersection over union for ``(left, top, right, bottom)`` boxes."""
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def window_process(hwnd, *, user32=None, kernel32=None):
    """Return the executable basename owning ``hwnd``, or ``None``."""
    user32 = user32 or ctypes.windll.user32
    kernel32 = kernel32 or ctypes.windll.kernel32
    process_id = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    if not process_id.value:
        return None
    open_process = kernel32.OpenProcess
    if hasattr(open_process, "restype"):
        open_process.restype = ctypes.c_void_p
    handle = open_process(0x1000, False, process_id.value)
    if not handle:
        return None
    try:
        size = ctypes.c_ulong(32768)
        path = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(
                handle, 0, path, ctypes.byref(size)):
            return None
        return os.path.basename(path.value) or None
    finally:
        kernel32.CloseHandle(handle)


def _nearest_monitor(rect, monitors):
    def score(item):
        bounds = item[1:5]
        overlap = (max(0, min(rect[2], bounds[2]) - max(rect[0], bounds[0])) *
                   max(0, min(rect[3], bounds[3]) - max(rect[1], bounds[1])))
        rx, ry = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        mx, my = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        return overlap, -((rx - mx) ** 2 + (ry - my) ** 2)
    return max(monitors, key=score)


def snapshot_layout(*, user32=None, windows_supplier=None, monitors=None,
                    process_resolver=None):
    """Capture visible, non-minimized windows on the current virtual desktop."""
    if windows_supplier is None:
        from commands import _windows_list
        windows_supplier = _windows_list
    user32 = user32 or ctypes.windll.user32
    available = list(monitors) if monitors is not None else enumerate_monitors(user32)
    if not available:
        return []
    process_resolver = process_resolver or (lambda hwnd: window_process(hwnd))
    # Tk's borderless windows normally have no title; keep exact names too for
    # builds/front ends that assign one explicitly.
    junk = {"program manager", "windows input experience", "flow overlay", "flow hud"}
    captured = []
    for hwnd, title in windows_supplier():
        title = str(title or "").strip()
        if not title or title.casefold() in junk or user32.IsIconic(hwnd):
            continue
        rect = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            continue
        box = (rect.left, rect.top, rect.right, rect.bottom)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        exe = process_resolver(hwnd) or ""
        monitor = _nearest_monitor(box, available)
        best_region, best_iou = None, -1.0
        for region in sorted(REGIONS):
            x, y, width, height = region_rect(monitor[1:5], region)
            score = _iou(box, (x, y, x + width, y + height))
            if score > best_iou:
                best_region, best_iou = region, score
        item = {"exe": exe, "title": title, "monitor": monitor[0]}
        if best_iou >= 0.55:
            item["region"] = best_region
        else:
            item.update({"do": "place_rect", "x": box[0] - monitor[1],
                         "y": box[1] - monitor[2], "w": box[2] - box[0],
                         "h": box[3] - box[1]})
        captured.append(item)
    return captured


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


def place_window_rect(title_query, monitor, x, y, width, height, *, user32=None,
                      windows_supplier=None, matcher=None, monitors=None):
    """Move a fuzzy-matched window to monitor-relative exact coordinates."""
    if windows_supplier is None or matcher is None:
        from commands import _match, _windows_list
        windows_supplier = windows_supplier or _windows_list
        matcher = matcher or _match
    user32 = user32 or ctypes.windll.user32
    titles = {title.lower(): (handle, title) for handle, title in windows_supplier()
              if title.strip()}
    key = matcher(title_query, titles)
    if not key:
        return None
    available = list(monitors) if monitors is not None else enumerate_monitors(user32)
    if not available:
        raise RuntimeError("Windows reported no monitors")
    target = next((item for item in available if item[0] == int(monitor)), None)
    if target is None:
        target = next((item for item in available if item[5]), available[0])
    handle, title = titles[key]
    if user32.IsIconic(handle):
        user32.ShowWindow(handle, 9)
    absolute_x, absolute_y = target[1] + int(x), target[2] + int(y)
    if not user32.MoveWindow(handle, absolute_x, absolute_y,
                             int(width), int(height), True):
        raise OSError("MoveWindow failed")
    return {"title": title, "monitor": target[0], "x": int(x), "y": int(y),
            "w": int(width), "h": int(height)}
