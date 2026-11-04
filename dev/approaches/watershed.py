"""Split-then-count screw counter (watershed approach).

count_screws(frame_bgr, unit=None) -> dict(count, items, debug)
calibrate(frame_bgr, true_count) -> unit (dict)

Pipeline (all at half resolution, S=2):
 1. Bin = largest bright blob (Otsu on blurred grey, closed, convex hull), eroded
    so the outer rim is ignored.  No hand-placed ROI.
 2. Background = heavy median of the frame (screws are thin, so a median window
    several screw-widths wide removes them).  d = grey - background.
    Polarity = sign of the larger extreme of d inside the bin (dark screws on a
    white floor -> negative, shiny screws on a grey floor -> positive).
    peak = 99.9th percentile of the polarity-signed contrast.
 3. Hysteresis mask.  strong = signed contrast > 0.5*peak (only real metal gets
    this far; ghost reflections in translucent walls stay at ~0.25*peak),
    weak = |d| > 0.25*peak.  Keep weak pixels connected to strong ones.  Both
    thresholds are fractions of the image's own peak contrast.  The weak level
    sits above the seam level: the thin gap between two touching screws keeps
    some floor light and drops below 0.25*peak, so the seam gets cut.
 4. Watershed split: distance transform of the mask; markers = DT maxima
    surviving an h-maxima test (h = 0.35*typical shank half-width), which cuts
    necks (end-to-end contacts, point contacts) but not the smooth ridge of one
    screw.  cv2.watershed then grows them inside the mask.
 5. Validation / counting.  Per instance: area, DT max (thickness).  Instances
    with thickness < 0.55x a single screw's (wall edge lines, text, glare
    lines) or area < 0.3 unit are dropped.  Each kept instance counts
    max(1, round(area/unit_area)), so a merged side-by-side pair with no
    visible seam still counts 2.
 unit (auto) = robust mode of instance areas (median, then median in
 [0.5,1.5]x); calibrated unit = kept mask area / true count.
"""
import cv2
import numpy as np

S = 2  # processing downscale


def _bin_mask(g):
    gb = cv2.GaussianBlur(g, (0, 0), 3)
    _, br = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, g.shape[1] // 38) | 1          # ~2.5% of width: closes screw holes in bin blob
    br = cv2.morphologyEx(br, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(br)
    if n < 2:
        return np.full_like(g, 255)
    i = 1 + int(np.argmax(st[1:, 4]))
    cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    m = np.zeros_like(g)
    cv2.drawContours(m, [hull], -1, 255, -1)
    e = max(3, g.shape[1] // 62) | 1          # drop the outer rim (~1.6% of width)
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))


def _floor_mask(g, binm):
    """Bin floor = the connected brightness class (Otsu on a smoothed image) that
    owns the bin centre.  Walls of a clear bin are a different brightness class
    from the floor; for an opaque bin both are in the same class and the floor
    mask is simply the whole bin."""
    h, w = g.shape
    s = 4
    sm = cv2.resize(g, (w // s, h // s), interpolation=cv2.INTER_AREA)
    bs = cv2.resize(binm, (w // s, h // s), interpolation=cv2.INTER_NEAREST)
    smooth = cv2.medianBlur(sm, 15)          # 15 px @1/8 full res = 120 px: wider than screws
    v = smooth[bs > 0]
    t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x, y, bw, bh = cv2.boundingRect(bs)
    c = smooth[y + bh // 3:y + 2 * bh // 3, x + bw // 3:x + 2 * bw // 3]
    fl = (smooth <= t) if np.mean(c <= t) > 0.5 else (smooth > t)
    fl = (fl & (bs > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fl)
    if n < 2:
        return binm
    i = 1 + int(np.argmax(st[1:, 4]))
    cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    m = np.zeros_like(sm)
    cv2.drawContours(m, [hull], -1, 255, -1)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(m, binm)


def _segment(frame_bgr):
    g0 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H0, W0 = g0.shape
    g = cv2.resize(g0, (W0 // S, H0 // S), interpolation=cv2.INTER_AREA)
    h, w = g.shape
    binm = _bin_mask(g)
    floor = _floor_mask(g, binm)
    inside = floor > 0
    gf = g.copy()
    gf[~inside] = int(np.median(g[inside]))      # outside floor must not leak into background
    # background: median at 1/4 working res, then inpaint pixels that are clearly
    # foreground and re-median, so a dense pile cannot drag the background toward itself.
    s = 4
    small = cv2.resize(gf, (w // s, h // s), interpolation=cv2.INTER_AREA)
    bgs = cv2.medianBlur(small, 21)
    for _ in range(2):
        dev = np.abs(small.astype(np.int16) - bgs.astype(np.int16))
        fg = (dev > max(8, 4 * np.median(dev))).astype(np.uint8)   # 4x the median abs deviation = clearly not floor
        fg = cv2.dilate(fg, np.ones((3, 3), np.uint8))
        bgs = cv2.medianBlur(cv2.inpaint(small, fg, 3, cv2.INPAINT_TELEA), 21)
    bg = cv2.resize(bgs, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    d = cv2.GaussianBlur(gf, (0, 0), 0.8).astype(np.float32) - bg
    dv = d[inside]
    pos, neg = np.percentile(dv, 99.9), -np.percentile(dv, 0.1)
    pol = 1.0 if pos > neg else -1.0
    peak = max(pos, neg, 1.0)
    sd = d * pol
    # Only polarity-matched contrast above 40% of the image's own peak counts as
    # metal.  Ghost reflections in translucent walls peak at ~20% (measured
    # 23-34 grey levels vs ~120 for real screws), shadows have opposite sign.
    m = ((sd > 0.4 * peak) & inside).astype(np.uint8) * 255
    # close specular stripes along shanks (2-3 px at working res)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    inv = cv2.bitwise_not(m)
    nh, hl, hst, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small_hole = np.zeros(nh, bool)
    small_hole[1:] = hst[1:, 4] < 100      # < ~400 px full res: socket holes / stripes, not floor gaps
    small_hole[hl[0, 0]] = False
    m[small_hole[hl]] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return dict(g=g, binm=floor, sd=sd, d=d, peak=peak, pol=pol, mask=m)


def _hmax_markers(dt, h):
    """Markers = regional maxima of dt with dynamic >= h (morphological reconstruction)."""
    seed = np.maximum(dt - h, 0).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    rec = seed
    for _ in range(200):
        nxt = np.minimum(cv2.dilate(rec, k), dt)
        if np.array_equal(nxt, rec):
            break
        rec = nxt
    return ((dt - rec) >= h - 1e-3) & (dt > 0)


def _split(mask):
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 1.5)
    half_w = float(np.median(dt[ridge])) if ridge.any() else 3.0   # typical shank half-width
    h = max(1.0, 0.35 * half_w)
    mk = _hmax_markers(dt, h).astype(np.uint8)
    nm, markers = cv2.connectedComponents(mk, connectivity=8)
    markers = markers.astype(np.int32)
    markers[mask == 0] = 0
    bgm = cv2.dilate(mask, np.ones((3, 3), np.uint8)) == 0
    markers[bgm] = nm  # background label
    rgb = cv2.cvtColor(np.clip(255 - dt * 10, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(rgb, markers)
    markers[mask == 0] = 0
    markers[markers == nm] = 0
    markers[markers < 0] = 0
    return markers, dt, half_w


def _instances(markers, dt):
    n = int(markers.max()) + 1
    area = np.bincount(markers.ravel(), minlength=n).astype(np.float64)
    dmax = np.zeros(n)
    np.maximum.at(dmax, markers.ravel(), dt.ravel())
    ys, xs = np.nonzero(markers)
    lab = markers[ys, xs]
    cx = np.bincount(lab, xs, n) / np.maximum(area, 1)
    cy = np.bincount(lab, ys, n) / np.maximum(area, 1)
    return [dict(id=i, area=area[i], dmax=dmax[i], cx=cx[i], cy=cy[i]) for i in range(1, n) if area[i] > 0]


def _robust_mode(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    m = np.median(v)
    for _ in range(3):
        sel = v[(v > 0.5 * m) & (v < 1.5 * m)]
        if len(sel) == 0:
            break
        m = np.median(sel)
    return float(m)


def _analyze(frame_bgr):
    seg = _segment(frame_bgr)
    markers, dt, half_w = _split(seg["mask"])
    inst = _instances(markers, dt)
    # thickness reference: robust mode of instance DT maxima, using only
    # instances that are not tiny specks
    big = [i for i in inst if i["dmax"] >= 2.0]
    thick = _robust_mode([i["dmax"] for i in big]) or 2.0
    cand = [i for i in inst if i["dmax"] >= 0.55 * thick]
    return seg, markers, dt, half_w, inst, cand, thick


def _count(cand, unit_area):
    items, total = [], 0
    for i in cand:
        r = i["area"] / unit_area
        if r < 0.3:
            continue
        k = max(1, int(round(r)))
        total += k
        items.append((int(i["cx"] * S), int(i["cy"] * S), k))
    return total, items


def calibrate(frame_bgr, true_count):
    seg, markers, dt, half_w, inst, cand, thick = _analyze(frame_bgr)
    tot = sum(i["area"] for i in cand)
    return dict(unit_area=tot / max(1, true_count))


def count_screws(frame_bgr, unit=None):
    seg, markers, dt, half_w, inst, cand, thick = _analyze(frame_bgr)
    if unit and unit.get("unit_area"):
        ua = unit["unit_area"]
    else:
        ua = _robust_mode([i["area"] for i in cand]) or 1.0
    total, items = _count(cand, ua)
    return dict(count=int(total), items=items,
                debug=dict(unit_area=ua, half_w=half_w, thick=thick, peak=seg["peak"], pol=seg["pol"],
                           n_inst=len(inst), n_cand=len(cand), mask=seg["mask"], markers=markers,
                           binm=seg["binm"]))


def overlay(frame_bgr, res):
    v = frame_bgr.copy()
    mk = res["debug"]["markers"]
    rng = np.random.default_rng(1)
    col = rng.integers(60, 255, (int(mk.max()) + 2, 3)).astype(np.uint8)
    col[0] = 0
    big = cv2.resize(mk.astype(np.uint16), (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    cm = col[big]
    sel = big > 0
    v[sel] = (0.45 * v[sel] + 0.55 * cm[sel]).astype(np.uint8)
    cs, _ = cv2.findContours(cv2.resize(res["debug"]["binm"], (v.shape[1], v.shape[0]), interpolation=0),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v, cs, -1, (0, 255, 0), 2)
    for x, y, k in res["items"]:
        cv2.circle(v, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(v, str(k), (x + 8, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(v, f"count={res['count']}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    return v
