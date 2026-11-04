"""Greedy rod fitting screw counter.

Pipeline (all parameters derived from the image or from the learned screw model):
  1. Bin + floor: Otsu on a 1/8 thumbnail -> largest bright blob = bin.  Objects are
     removed from the thumbnail by a morphological open/close (polarity picked
     automatically), then the floor is flood-filled from the bin centre with a
     small gradient tolerance.  The floor region = minAreaRect of the flood,
     eroded by one screw width (removes the floor/wall seam line).
  2. Background = normalised-convolution interpolation of floor pixels only, so
     screws don't bias it.  Signed contrast c = (gray - bg) * polarity.
  3. Foreground threshold = 0.5 * typical foreground contrast (median contrast of
     pixels clearly above noise).  Faint ghost reflections in translucent walls
     have roughly 1/3 of a real screw's contrast so they fall below it.
  4. Screw model (L, w): iterate the median area over elongated components,
     keeping only components near the lower mode (single screws).
  5. Each component is explained greedily by L x w oriented rods (matched filter
     over angles, step atan(w/L)); a one-sided flank / end penalty makes rods snap
     to edges of side-by-side or end-to-end bundles.  Stop when the best rod
     covers < half a screw of unexplained pixels.
"""
import cv2
import numpy as np

S = 2          # processing downscale for the mask (1080p -> 540p)
TH = 8         # thumbnail downscale for bin/floor geometry


def _otsu(v):
    v = np.clip(v, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _floor(gray):
    H, W = gray.shape
    sm = cv2.resize(gray, (W // TH, H // TH), interpolation=cv2.INTER_AREA)
    _, br = cv2.threshold(cv2.GaussianBlur(sm, (0, 0), 1), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, cen = cv2.connectedComponentsWithStats(br)
    i = 1 + int(np.argmax(st[1:, 4]))
    x, y, bw, bh = st[i, :4]
    binm = (lab == i)
    # polarity: sign of strongest local deviations in the central half of the bin
    med = cv2.medianBlur(sm, 9).astype(np.float32)
    d = sm.astype(np.float32) - med
    cb = np.zeros(sm.shape, bool)
    cb[y + bh // 4:y + 3 * bh // 4, x + bw // 4:x + 3 * bw // 4] = True
    sel = (np.abs(d) > 20) & cb
    pol = 1.0 if d[sel].sum() > 0 else -1.0
    # remove objects (thin at this scale) -> smooth floor, flood from centre
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(sm, cv2.MORPH_OPEN if pol > 0 else cv2.MORPH_CLOSE, k)
    clean = cv2.medianBlur(clean, 5)
    ff = np.zeros((sm.shape[0] + 2, sm.shape[1] + 2), np.uint8)
    seed = (int(cen[i][0]), int(cen[i][1]))
    cv2.floodFill(clean.copy(), ff, seed, 0, 2, 2,
                  4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    fl = (ff[1:-1, 1:-1] > 0) & binm
    pts = np.argwhere(fl)[:, ::-1].astype(np.float32)
    rect = cv2.minAreaRect(pts)
    # floor samples for the background model: flooded & no local object
    samp = (fl & (np.abs(d) < 15)).astype(np.float32)
    sig = bw / 8.0
    num = cv2.GaussianBlur(sm.astype(np.float32) * samp, (0, 0), sig)
    den = cv2.GaussianBlur(samp, (0, 0), sig)
    bg = num / np.maximum(den, 1e-3)
    return pol, rect, bg


def _mask(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    pol, rect, bg = _floor(gray)
    g = cv2.resize(gray, (W // S, H // S), interpolation=cv2.INTER_AREA).astype(np.float32)
    bgS = cv2.resize(bg, (W // S, H // S), interpolation=cv2.INTER_LINEAR)
    c = (g - bgS) * pol
    f = TH / S
    (cx, cy), (rw, rh), ang = rect
    floor = np.zeros(g.shape, np.uint8)
    box = cv2.boxPoints(((cx * f, cy * f), (rw * f, rh * f), ang))
    cv2.fillPoly(floor, [box.astype(np.int32)], 255)
    v = c[floor > 0]
    sigma = 1.4826 * np.median(np.abs(v - np.median(v))) + 1.0   # +1: quantisation
    strong = v[v > 8 * sigma]
    typ = float(np.median(strong)) if strong.size > 50 else 16 * sigma
    thr = max(4 * sigma, 0.5 * typ)
    m = ((c > thr) & (floor > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return m, floor, dict(pol=pol, sigma=sigma, typ=typ, thr=thr, box=box)


def _components(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return n, lab, st


def _learn_model(m):
    n, lab, st = _components(m)
    cands = []
    for i in range(1, n):
        a = st[i, 4]
        if a < 20:
            continue
        ys, xs = np.nonzero(lab[st[i, 1]:st[i, 1] + st[i, 3], st[i, 0]:st[i, 0] + st[i, 2]] == i)
        (_, _), (a1, a2), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        L, w = max(a1, a2), max(1.0, min(a1, a2))
        fill = a / max(1.0, L * w)
        if L / w >= 1.5 and fill > 0.5:
            cands.append((a, L, a / L))
    if not cands:
        return None
    cands = np.array(cands)
    A = np.median(cands[:, 0])
    for _ in range(5):        # move to the lower mode (single screws)
        sel = cands[(cands[:, 0] < 1.5 * A) & (cands[:, 0] > 0.5 * A)]
        if len(sel) == 0:
            break
        A2 = np.median(sel[:, 0])
        if A2 == A:
            break
        A = A2
    sel = cands[(cands[:, 0] < 1.5 * A) & (cands[:, 0] > 0.5 * A)]
    return dict(L=float(np.median(sel[:, 1])), w=float(np.median(sel[:, 2])), n_model=int(len(sel)))


def _kernels(L, w, step_deg):
    """Per angle: rod kernel + 4 one-sided flank kernels (left/right sides, two ends)."""
    ks = []
    R = int(np.ceil(np.hypot(L + 2 * w, 3 * w) / 2)) + 1
    sz = 2 * R + 1
    for a in np.arange(0, 180, step_deg):
        t = np.deg2rad(a)
        def rect(cx, cy, l, ww):
            img = np.zeros((sz, sz), np.float32)
            b = cv2.boxPoints(((R + cx, R + cy), (l, ww), a))
            cv2.fillConvexPoly(img, np.round(b * 4).astype(np.int32), 1.0, cv2.LINE_8, 2)
            return img
        u = np.array([np.cos(t), np.sin(t)]); nrm = np.array([-u[1], u[0]])
        rod = rect(0, 0, L, w)
        fw = max(1.0, w / 2)
        s1 = rect(*(nrm * (w + fw) / 2), L, fw)
        s2 = rect(*(-nrm * (w + fw) / 2), L, fw)
        e1 = rect(*(u * (L + fw) / 2), fw, w)
        e2 = rect(*(-u * (L + fw) / 2), fw, w)
        for k in (s1, s2, e1, e2):
            k[rod > 0] = 0
        ks.append((a, rod, s1, s2, e1, e2))
    return ks, R


def _fit_component(res, orig, ks, R, L, w, min_cov):
    """Greedy rods on one component ROI. res: residual (float32 0/1), orig: original mask."""
    rods = []
    pad = R
    resp = np.pad(res, pad)
    org = np.pad(orig, pad)
    lam = 0.5
    while True:
        best = None
        for (a, rod, s1, s2, e1, e2) in ks:
            cov = cv2.filter2D(resp, -1, rod, borderType=cv2.BORDER_CONSTANT)
            if cov.max() < min_cov:
                continue
            # flank penalty: at least one side and one end should be "free"
            # (background or already explained by an earlier rod)
            free = resp  # unexplained foreground counts as occupied
            fs = np.minimum(cv2.filter2D(free, -1, s1, borderType=cv2.BORDER_CONSTANT) / max(s1.sum(), 1),
                            cv2.filter2D(free, -1, s2, borderType=cv2.BORDER_CONSTANT) / max(s2.sum(), 1))
            fe = np.minimum(cv2.filter2D(free, -1, e1, borderType=cv2.BORDER_CONSTANT) / max(e1.sum(), 1),
                            cv2.filter2D(free, -1, e2, borderType=cv2.BORDER_CONSTANT) / max(e2.sum(), 1))
            score = cov * (1 - lam * fs) * (1 - lam * fe)
            score[cov < min_cov] = -1
            j = int(np.argmax(score))
            sc = score.flat[j]
            if sc > 0 and (best is None or sc > best[0]):
                yy, xx = divmod(j, score.shape[1])
                best = (sc, a, xx, yy, cov.flat[j])
        if best is None:
            break
        sc, a, xx, yy, cv = best
        rods.append((xx - pad, yy - pad, a))
        b = cv2.boxPoints(((float(xx), float(yy)), (L + 1, w + 1), a))
        cv2.fillConvexPoly(resp, np.round(b).astype(np.int32), 0.0)
        if len(rods) > 200:
            break
    return rods


def count_screws(frame_bgr, unit=None):
    m, floor, dbg = _mask(frame_bgr)
    model = _learn_model(m)
    if unit is not None:
        model = dict(model or {}, L=unit['L'], w=unit['w'])
    if model is None:
        return dict(count=0, items=[], debug=dict(dbg, mask=m))
    L, w = model['L'], model['w']
    step = max(3.0, np.rad2deg(np.arctan2(w, L)))
    ks, R = _kernels(L, w, step)
    min_cov = 0.5 * L * w     # a rod must explain at least half a screw
    n, lab, st = _components(m)
    items = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < 0.5 * min_cov:
            continue
        sub = (lab[y:y + bh, x:x + bw] == i).astype(np.float32)
        rods = _fit_component(sub.copy(), sub, ks, R, L, w, min_cov)
        for (rx, ry, ra) in rods:
            items.append((float((rx + x) * S), float((ry + y) * S), float(ra)))
    dbg.update(model=model, L=L, w=w, step=step, mask=m, floor=floor)
    return dict(count=len(items), items=items, debug=dbg)


def calibrate(frame_bgr, true_count):
    m, _, _ = _mask(frame_bgr)
    model = _learn_model(m)
    L0, w0 = model['L'], model['w']
    hits = []
    best = None
    for s in np.arange(0.7, 1.31, 0.05):
        u = dict(L=L0 * s, w=w0)
        c = count_screws(frame_bgr, u)['count']
        err = abs(c - true_count)
        if best is None or err < best[0]:
            best = (err, s)
        if err == 0:
            hits.append(s)
    s = float(np.median(hits)) if hits else best[1]
    return dict(L=L0 * s, w=w0, scale=s)


def draw(frame_bgr, res):
    v = frame_bgr.copy()
    d = res['debug']
    if 'box' in d:
        cv2.polylines(v, [(d['box'] * S).astype(np.int32)], True, (0, 255, 0), 2)
    if 'mask' in d:
        mk = cv2.resize(d['mask'], (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST)
        v[mk > 0] = (0.6 * v[mk > 0] + 0.4 * np.array([255, 128, 0])).astype(np.uint8)
    L, w = d.get('L', 0) * S, d.get('w', 0) * S
    for k, (x, y, a) in enumerate(res['items']):
        b = cv2.boxPoints(((x, y), (L, w), a)).astype(np.int32)
        cv2.polylines(v, [b], True, (0, 0, 255), 2)
        cv2.putText(v, str(k + 1), (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(v, 'count=%d' % res['count'], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
    return v
