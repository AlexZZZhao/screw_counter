"""LINE-SEGMENT screw counter.

Idea: every screw is a straight rigid rod of (nearly) constant length L and shank
width W.  We detect straight edges with LSD, turn each edge into rod-centreline
hypotheses, verify them against a polarity-agnostic foreground mask, merge
collinear pieces into rods, suppress duplicates greedily (each accepted rod
"consumes" a W-wide band), and finally count  sum(max(1, round(len_i / L))).

Two touching screws side-by-side -> two distinct parallel centrelines ~W apart,
each with its own band -> 2 rods.  End-to-end -> one collinear rod of ~2L -> 2.

Everything runs at half resolution (960x540 for a 1080p frame).

Public API
    count_screws(frame_bgr, unit=None) -> dict(count, items, debug)
    calibrate(frame_bgr, true_count) -> unit   (unit = dict(L=...) in half-res px)
"""
import cv2
import numpy as np

SCALE = 0.5            # processing scale (speed); all px below are at this scale
_LSD = None


def _lsd():
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    return _LSD


# ----------------------------------------------------------------- bin / floor
def _bin_mask(g):
    """Largest bright blob (the bin) on the dark table -> convex hull.
    Same idea as dev/proto3.py."""
    gb = cv2.GaussianBlur(g, (0, 0), 1.5)
    _, bright = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(round(g.shape[1] / 40)) | 1)          # close gaps ~2.5% of width
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    if n < 2:
        return np.full_like(g, 255)
    i = 1 + int(np.argmax(st[1:, 4]))
    cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    m = np.zeros_like(g)
    cv2.drawContours(m, [hull], -1, 255, -1)
    e = max(3, k // 2 | 1)                                  # drop the rim itself
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))


def _floor_mask(bg, binm):
    """Flood-fill the smooth background (median-filtered, screws removed) from the
    bin centre with a tiny tolerance (2 grey levels per step).  On the clear bin the
    floor/wall junction is a step in brightness so the fill stops at the floor; on
    the white bin walls and floor are the same colour so the fill takes both (the
    wall ghosts are then rejected by contrast)."""
    small = cv2.resize(bg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    bs = cv2.resize(binm, small.shape[::-1], interpolation=cv2.INTER_NEAREST)
    im = small.copy()
    im[bs == 0] = 0
    ys, xs = np.nonzero(bs)
    if len(xs) == 0:
        return binm
    cx, cy = int(xs.mean()), int(ys.mean())
    # seed at the most typical floor value near the centre
    mk = np.zeros((im.shape[0] + 2, im.shape[1] + 2), np.uint8)
    cv2.floodFill(im, mk, (cx, cy), 255, 2, 2,
                  4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    fl = mk[1:-1, 1:-1]
    if fl.sum() < 0.2 * bs.sum():          # fill failed (seed on a screw?) -> bin
        return binm
    cs, _ = cv2.findContours(fl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(cs))
    out = np.zeros_like(fl)
    cv2.drawContours(out, [hull], -1, 255, -1)
    out = cv2.resize(out, bg.shape[::-1], interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(out, binm)


# ------------------------------------------------------------- foreground mask
def _foreground(g, floor):
    """|gray - smooth background| ; background = median filter much larger than a
    screw width.  Threshold = Otsu over the in-floor pixels that differ from the
    background by more than sensor noise, so it adapts to the scene contrast and
    sits between faint wall ghosts/glare and real metal."""
    H, W = g.shape
    sm = cv2.resize(g, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(sm, 15), (W, H), interpolation=cv2.INTER_LINEAR)
    d = cv2.absdiff(g, bg)
    inside = floor > 0
    v = d[inside]
    noise = max(4.0, 3.0 * float(np.median(v)))          # ~3x typical floor deviation
    vv = v[v > noise]
    if len(vv) < 50:
        T = 255.0
    else:
        T, _ = cv2.threshold(vv.reshape(-1, 1), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # hysteresis: weak pixels (> T/2) survive only if their component holds a
    # strong (> T) pixel; this recovers shank highlights that happen to match the
    # floor brightness without admitting faint ghosts (which have no strong pixel).
    strong = (d > T) & inside
    weak = ((d > 0.5 * T) & inside).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[strong])] = True; keep[0] = False
    fg = keep[lab].astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # fill enclosed holes smaller than ~a screw head (specular stripes inside a shank)
    inv = cv2.bitwise_not(fg)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = np.zeros(n, bool); small[1:] = st[1:, 4] < 0.0005 * fg.size
    fg[small[lab]] = 255
    return fg, d, bg, float(T)


def _estimate_width(fg):
    """Shank width W = 2 x typical distance-transform ridge value."""
    dt = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 1.0)
    v = dt[ridge]
    if len(v) < 20:
        return 4.0
    return float(np.clip(2.0 * np.median(v), 3.0, 40.0))


# ------------------------------------------------------------------ rod logic
def _runs(ok, max_gap):
    """Longest run of True allowing gaps up to max_gap samples. returns (i0, i1)."""
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return (0, -1)
    brk = np.flatnonzero(np.diff(idx) - 1 > max_gap)
    starts = np.r_[0, brk + 1]; ends = np.r_[brk, len(idx) - 1]
    k = int(np.argmax(idx[ends] - idx[starts]))
    return int(idx[starts[k]]), int(idx[ends[k]])


def _verify(p0, p1, fg, W):
    """Trim a centreline hypothesis to the part that is actually covered by
    foreground across most of a W-wide band.  Returns new (p0, p1) or None."""
    d = p1 - p0
    Lseg = float(np.hypot(*d))
    if Lseg < 1:
        return None
    u = d / Lseg
    n = np.array([-u[1], u[0]])
    ext = W                                    # also search a bit past the ends
    ts = np.arange(-ext, Lseg + ext + 1, 1.0)
    offs = np.array([-0.3, 0.0, 0.3]) * W      # inner 60% of the shank width
    pts = p0[None, None, :] + ts[:, None, None] * u + offs[None, :, None] * n
    xi = np.clip(np.round(pts[..., 0]).astype(int), 0, fg.shape[1] - 1)
    yi = np.clip(np.round(pts[..., 1]).astype(int), 0, fg.shape[0] - 1)
    cov = (fg[yi, xi] > 0).mean(axis=1)
    i0, i1 = _runs(cov >= 0.99, max_gap=int(W))  # all of the inner 60% on metal
    if i1 < 0:
        return None
    a, b = ts[i0], ts[i1]
    return p0 + a * u, p0 + b * u


def _merge_collinear(rods, W):
    """Union-find merge of centrelines that are parallel (<6 deg), on the same line
    (perpendicular offset < W/2) and touch/overlap along the axis (gap < W)."""
    n = len(rods)
    par = list(range(n))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    P0 = np.array([r[0] for r in rods]); P1 = np.array([r[1] for r in rods])
    D = P1 - P0
    Ls = np.hypot(D[:, 0], D[:, 1]) + 1e-6
    U = D / Ls[:, None]
    cos_t = np.cos(np.radians(6))
    for i in range(n):
        dots = np.abs(U @ U[i])
        cand = np.nonzero(dots > cos_t)[0]
        for j in cand:
            if j <= i:
                continue
            ni = np.array([-U[i, 1], U[i, 0]])
            # perpendicular offsets of j's endpoints from i's line
            o0 = (P0[j] - P0[i]) @ ni; o1 = (P1[j] - P0[i]) @ ni
            if max(abs(o0), abs(o1)) > 0.3 * W:   # < neighbour spacing W/2
                continue
            t0 = (P0[j] - P0[i]) @ U[i]; t1 = (P1[j] - P0[i]) @ U[i]
            lo, hi = min(t0, t1), max(t0, t1)
            gap = max(lo - Ls[i], -hi, 0)
            if gap > W:
                continue
            par[f(i)] = f(j)
    groups = {}
    for i in range(n):
        groups.setdefault(f(i), []).append(i)
    out = []
    for idx in groups.values():
        pts = np.vstack([P0[idx], P1[idx]]).astype(np.float64)
        w = np.repeat(Ls[idx], 1)
        # direction: length-weighted average of member directions (sign aligned)
        ref = U[idx[int(np.argmax(Ls[idx]))]]
        uu = np.array([U[k] if U[k] @ ref >= 0 else -U[k] for k in idx])
        u = (uu * w[:, None]).sum(0); u /= np.linalg.norm(u)
        nn = np.array([-u[1], u[0]])
        mids = (P0[idx] + P1[idx]) / 2
        c_off = np.average(mids @ nn, weights=w)
        t = pts @ u
        a, b = t.min(), t.max()
        base = c_off * nn
        out.append((base + a * u, base + b * u, float(Ls[idx].sum())))
    return out


def _band_points(p0, p1, W, frac=0.8):
    d = p1 - p0
    L = float(np.hypot(*d)) + 1e-6
    u = d / L; n = np.array([-u[1], u[0]])
    ts = np.arange(0, L + 1, 1.0)
    offs = np.linspace(-frac / 2, frac / 2, 5) * W
    pts = p0 + ts[:, None, None] * u + offs[None, :, None] * n
    return pts.reshape(-1, 2)


# ------------------------------------------------------------------- pipeline
def _detect_rods(frame_bgr):
    g0 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g0, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    binm = _bin_mask(g)
    fg, diff, bg, T = _foreground(g, binm)
    floor = _floor_mask(bg, binm)
    W = _estimate_width(cv2.bitwise_and(fg, floor))
    # pull the floor boundary in by 1.5 shank widths: floor/wall junction lines and
    # rim glare are long straight edges hugging the boundary.  A screw touching the
    # wall loses only its outer ~W, which the length rounding tolerates.
    e = int(2 * round(1.5 * W) + 1)
    floor = cv2.erode(floor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
    fg[floor == 0] = 0

    segs = _lsd().detect(g)[0]
    segs = np.zeros((0, 4), np.float32) if segs is None else segs.reshape(-1, 4)
    seglen = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs = segs[seglen >= 1.5 * W]            # shorter than 1.5 widths: head rims, noise

    # every edge gives 2 centreline hypotheses, W/2 to either side.  For two
    # shanks touching side by side (outer edges at 0 and 2W, contact line at W)
    # these land exactly on the two true centres (W/2, 3W/2); the wrong-side
    # hypotheses fall off the metal and are trimmed away by _verify.
    cands = []
    for x1, y1, x2, y2 in segs:
        p0 = np.array([x1, y1], np.float64); p1 = np.array([x2, y2], np.float64)
        d = p1 - p0; u = d / (np.hypot(*d) + 1e-9); n = np.array([-u[1], u[0]])
        for s in (-0.5, 0.5):
            r = _verify(p0 + s * W * n, p1 + s * W * n, fg, W)
            if r is not None and np.hypot(*(r[1] - r[0])) >= 2.0 * W:
                cands.append(r)
    rods = _merge_collinear(cands, W) if cands else []

    # greedy NMS: longest first; a rod is kept if >50% of its band is still unused
    # order by edge support (summed length of the hypotheses that merged into the
    # rod): a true centre is backed by edges on both of its flanks.
    rods.sort(key=lambda r: -r[2])
    used = np.zeros(g.shape, np.uint8)
    kept = []
    for p0, p1, k in rods:
        pts = _band_points(p0, p1, W)
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, g.shape[1] - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, g.shape[0] - 1)
        fresh = 1.0 - (used[yi, xi] > 0).mean()
        if fresh < 0.5:
            continue
        kept.append((p0, p1))
        cv2.line(used, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                 255, max(1, int(round(1.5 * W))))
        # 1.5W-thick claim: a duplicate offset by W/2 finds ~80% of its band used
        # and is dropped; a touching neighbour (centre W away) finds only ~20%.
    lens = np.array([np.hypot(*(b - a)) for a, b in kept]) if kept else np.zeros(0)
    return kept, lens, dict(W=W, T=T, n_segs=int(len(segs)), n_cands=len(cands),
                            fg=fg, floor=floor)


def _auto_L(lens):
    """Screw length estimate without calibration: the longest rods are the most
    likely to be whole un-occluded screws, but a few may be 2 end-to-end.  Take the
    median of the rods in the upper half of the length distribution."""
    if len(lens) == 0:
        return 1.0
    s = np.sort(lens)
    return float(np.median(s[len(s) // 2:]))


def _count_from(lens, L, maxk=4):
    # a rod shorter than L/2 is a fragment of something already counted (head rim,
    # piece of a crossing screw, wall edge) -> 0.  Under the 'touching, not
    # overlapping' rule a real screw always shows > L/2 of its length.
    k = np.round(lens / L)
    k[k > maxk] = 0            # far longer than any screw train -> structure, drop
    return k.astype(int)


def count_screws(frame_bgr, unit=None):
    rods, lens, dbg = _detect_rods(frame_bgr)
    L = unit['L'] if unit else _auto_L(lens)
    k = _count_from(lens, L)
    inv = 1.0 / SCALE
    items = []
    for (p0, p1), ki in zip(rods, k):
        if ki == 0:
            continue
        items.append(dict(p0=tuple((p0 * inv).round(1)), p1=tuple((p1 * inv).round(1)),
                          n=int(ki)))
    dbg.update(L=L, lens=lens.tolist(), per_rod=k.tolist(), rods=rods)
    return dict(count=int(k.sum()), items=items, debug=dbg)


def calibrate(frame_bgr, true_count):
    """unit = screw length L that reproduces true_count on this frame.  Scan L over
    the observed rod lengths and take the middle of the interval that matches
    (or the closest-count interval)."""
    rods, lens, dbg = _detect_rods(frame_bgr)
    if len(lens) == 0:
        return dict(L=1.0)
    Ls = np.linspace(0.3 * lens.max(), 1.2 * lens.max(), 400)
    errs = np.array([abs(_count_from(lens, L).sum() - true_count) for L in Ls])
    best = errs.min()
    ok = Ls[errs == best]
    return dict(L=float(np.median(ok)), W=dbg['W'], residual=int(best))
