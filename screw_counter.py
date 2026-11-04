"""Live screw counter for a downward-facing USB camera.

Run:  python screw_counter.py            (auto-picks the USB camera)
      python screw_counter.py --camera 1
      python screw_counter.py --image dev/frame.png   (offline test)

Keys (in the video window):
  r  select counting region (drag a box over the bin floor, ENTER to accept)
  k  calibrate: type the true count in the terminal (do this once per screw type)
  u  forget calibration (go back to auto estimate)
  f  refocus (camera re-runs autofocus, then locks)
  c  switch to next camera
  [ ]  lower / raise detection threshold
  m  toggle mask view
  s  save snapshot
  q / ESC  quit
"""
import argparse
import json
import platform
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).with_name("screw_counter_config.json")

DEFAULTS = {
    "roi": None,            # [x, y, w, h] in full-frame pixels
    "unit_area": None,      # pixel area of one screw; None = estimate each frame
    "min_thresh": 25,       # floor for Otsu threshold on top-hat image
    "thresh_offset": 0,     # user adjustment via [ ]
    "width": 1920,
    "height": 1080,
}


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except (OSError, ValueError):
            print(f"warning: could not read {CONFIG_PATH}, using defaults")
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------- camera

def backend():
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def probe_cameras(max_index=6):
    """Return indices that open and deliver a frame."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend())
        if cap.isOpened() and cap.read()[0]:
            found.append(i)
        cap.release()
    return found


def camera_names():
    """Best-effort device names (Windows only, needs pygrabber)."""
    if platform.system() != "Windows":
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return None


def pick_camera(available):
    """Prefer a camera whose name doesn't look built-in; else the highest index.

    Built-in laptop webcams almost always enumerate first, so an external USB
    camera is usually the last index.
    """
    names = camera_names()
    if names:
        builtin = ("integrated", "built-in", "facetime", "internal", "hd webcam", "ir camera")
        external = [i for i in available
                    if i < len(names) and not any(b in names[i].lower() for b in builtin)]
        if external:
            return external[0]
    return available[-1]


def open_camera(index, cfg):
    cap = cv2.VideoCapture(index, backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    return cap


def sharpness(frame, roi=None):
    if roi:
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


class FocusWaiter:
    """Holds off counting until autofocus settles: sharpness stable for ~10
    frames, or a timeout. Autofocus is left on - some cameras (e.g. Razer Kiyo
    Pro) drop focus when it's disabled and don't report the lens position."""

    def __init__(self, cap, timeout=8.0):
        self.cap, self.timeout = cap, timeout
        self.start = time.time()
        self.hist = deque(maxlen=10)
        self.ready = False

    def update(self, frame, roi):
        if self.ready:
            return True
        self.hist.append(sharpness(frame, roi))
        elapsed = time.time() - self.start
        stable = (len(self.hist) == self.hist.maxlen and elapsed > 1.5
                  and (max(self.hist) - min(self.hist)) < 0.03 * max(self.hist))
        if stable or elapsed > self.timeout:
            self.ready = True
        return self.ready

    def refocus(self):
        # Toggling autofocus makes most UVC cameras re-run their focus search.
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.__init__(self.cap, self.timeout)


# ---------------------------------------------------------------- detection

def detect(frame, cfg):
    """Return (count, blobs, mask, unit). blobs = [(x, y, w, h, n_screws, area)] in frame coords."""
    H, W = frame.shape[:2]
    rx, ry, rw, rh = cfg["roi"] or (0, 0, W, H)
    roi = frame[ry:ry + rh, rx:rx + rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    scale = H / 1080.0
    k = max(21, int(61 * scale)) | 1  # must be wider than a screw head
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    otsu, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu, cfg["min_thresh"]) + cfg["thresh_offset"]
    _, mask = cv2.threshold(tophat, thr, 255, cv2.THRESH_BINARY)
    ck = max(3, int(9 * scale)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))

    # Filled outer contours: screw heads have dark hex sockets, filling them
    # makes area proportional to screw count.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 150 * scale * scale
    comps = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        touches_edge = x <= 2 or y <= 2 or x + w >= rw - 2 or y + h >= rh - 2
        if touches_edge and max(w, h) / max(1, min(w, h)) > 4:  # bin wall / rim glare
            continue
        comps.append((x + rx, y + ry, w, h, a))

    unit = cfg["unit_area"] or estimate_unit(comps)
    blobs = []
    for x, y, w, h, a in comps:
        if unit and a < 0.3 * unit:  # glint or fragment
            continue
        c = max(1, round(a / unit)) if unit else 1
        blobs.append((x, y, w, h, c, a))
    # Sum area first, then divide: avoids per-blob rounding error piling up.
    total = round(sum(b[5] for b in blobs) / unit) if unit else len(blobs)
    return total, blobs, mask, unit


def estimate_unit(comps):
    """Typical single-screw area.

    Fragments (glints, partial shanks) are small and clumps are large, so take
    the median, drop blobs well below it, and use the median of what's left
    below the upper quartile.
    """
    if not comps:
        return None
    areas = sorted(c[4] for c in comps)
    med = statistics.median(areas)
    core = [a for a in areas if 0.5 * med <= a <= 1.6 * med] or areas
    return statistics.median(core)


# ---------------------------------------------------------------- UI

def draw(frame, count, blobs, cfg, info):
    out = frame.copy()
    if cfg["roi"]:
        x, y, w, h = cfg["roi"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 2)
    for x, y, w, h, c, _ in blobs:
        color = (0, 220, 0) if c == 1 else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if c > 1:
            cv2.putText(out, f"x{c}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    s = out.shape[0] / 1080.0
    cv2.rectangle(out, (0, 0), (int(520 * s), int(150 * s)), (0, 0, 0), -1)
    cv2.putText(out, f"Screws: {count}", (int(15 * s), int(85 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4 * s, (255, 255, 255), max(2, int(5 * s)))
    cv2.putText(out, info, (int(15 * s), int(130 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (200, 200, 200), max(1, int(2 * s)))
    return out


def fit_to_screen(img, max_w=1600, max_h=900):
    h, w = img.shape[:2]
    f = min(max_w / w, max_h / h, 1.0)
    return (cv2.resize(img, (int(w * f), int(h * f))), f) if f < 1 else (img, 1.0)


def select_roi(frame):
    disp, f = fit_to_screen(frame)
    r = cv2.selectROI("Select bin floor, ENTER to accept, C to cancel", disp, showCrosshair=False)
    cv2.destroyWindow("Select bin floor, ENTER to accept, C to cancel")
    if r[2] < 10 or r[3] < 10:
        return None
    return [int(v / f) for v in r]


def main():
    ap = argparse.ArgumentParser(description="Count screws seen by a USB camera.")
    ap.add_argument("--camera", type=int, help="camera index (default: auto-detect USB cam)")
    ap.add_argument("--image", help="run on a still image instead of a camera")
    ap.add_argument("--list", action="store_true", help="list cameras and exit")
    ap.add_argument("--once", action="store_true", help="print count and exit (with --image)")
    args = ap.parse_args()

    cfg = load_config()

    if args.list:
        names = camera_names() or []
        for i in probe_cameras():
            print(i, names[i] if i < len(names) else "")
        return

    cap, cams, cam_i = None, [], 0
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            sys.exit(f"could not read {args.image}")
        if args.once:
            count, blobs, _, unit = detect(still, cfg)
            print(count)
            return
    else:
        cams = probe_cameras()
        if not cams:
            sys.exit("No camera found. Is the USB camera plugged in?")
        cam = args.camera if args.camera is not None else pick_camera(cams)
        cam_i = cams.index(cam) if cam in cams else 0
        cap = open_camera(cam, cfg)
        print(f"cameras: {cams}, using {cam}")

    history = deque(maxlen=15)
    area_hist = deque(maxlen=15)
    focus = FocusWaiter(cap) if cap is not None else None
    show_mask = False
    win = "Screw Counter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    fps = 0.0

    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed, reopening...")
                cap.release()
                time.sleep(0.5)
                cap = open_camera(cams[cam_i], cfg)
                focus = FocusWaiter(cap)
                continue
        else:
            frame = still.copy()

        if focus is not None and not focus.update(frame, cfg["roi"]):
            out = frame.copy()
            s = out.shape[0] / 1080.0
            cv2.putText(out, "Focusing...", (int(30 * s), int(90 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.4 * s, (0, 255, 255), max(2, int(5 * s)))
            cv2.imshow(win, fit_to_screen(out)[0])
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue

        raw, blobs, mask, unit = detect(frame, cfg)
        history.append(raw)
        area_hist.append(sum(b[5] for b in blobs))
        count = int(round(statistics.median(history)))

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-6, now - t_prev)
        t_prev = now
        mode = "calibrated" if cfg["unit_area"] else "auto"
        info = (f"unit {unit:.0f}px ({mode})  thr {cfg['thresh_offset']:+d}  {fps:.0f} fps"
                if unit else f"thr {cfg['thresh_offset']:+d}  {fps:.0f} fps")

        if show_mask:
            view = np.zeros(frame.shape[:2], np.uint8)
            x, y, w, h = cfg["roi"] or (0, 0, frame.shape[1], frame.shape[0])
            view[y:y + h, x:x + w] = mask
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        else:
            view = frame
        cv2.imshow(win, fit_to_screen(draw(view, count, blobs, cfg, info))[0])

        key = cv2.waitKey(1 if cap is not None else 30) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord("r"):
            roi = select_roi(frame)
            if roi:
                cfg["roi"] = roi
                cfg["unit_area"] = None  # region changed; recalibrate
                save_config(cfg)
                history.clear()
                area_hist.clear()
        elif key == ord("k"):
            try:
                true_n = int(input(f"Showing {count}. Enter the true count: "))
            except ValueError:
                true_n = 0
            area = statistics.median(area_hist) if area_hist else 0
            if true_n > 0 and area:
                cfg["unit_area"] = area / true_n
                save_config(cfg)
                history.clear()
                print(f"calibrated: one screw = {cfg['unit_area']:.0f} px")
        elif key == ord("u"):
            cfg["unit_area"] = None
            save_config(cfg)
            history.clear()
        elif key == ord("["):
            cfg["thresh_offset"] -= 3
            save_config(cfg)
            history.clear()
        elif key == ord("]"):
            cfg["thresh_offset"] += 3
            save_config(cfg)
            history.clear()
        elif key == ord("m"):
            show_mask = not show_mask
        elif key == ord("f") and focus is not None:
            focus.refocus()
            history.clear()
        elif key == ord("c") and len(cams) > 1:
            cap.release()
            cam_i = (cam_i + 1) % len(cams)
            cap = open_camera(cams[cam_i], cfg)
            focus = FocusWaiter(cap)
            history.clear()
            print(f"switched to camera {cams[cam_i]}")
        elif key == ord("s"):
            name = time.strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, draw(frame, count, blobs, cfg, info))
            print(f"saved {name}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
