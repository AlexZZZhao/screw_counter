"""Skeleton / width-geometry screw counter.

Pipeline (all sizes derived from the image; analysis at half resolution):
 1. Bin = largest Otsu-bright blob (convex hull) of a 1/4-scale grey image.
 2. Floor = the Otsu class of the median-smoothed bin interior that contains
    the bin centre (connected to it).  For a clear bin this cuts off the
    bright walls; for a white bin floor and walls fall in the same class, so
    the whole bin is kept and ghosts are rejected by contrast (step 4).
 3. Polarity: sign of the strongest 2% deviations from a coarse median
    background on the floor -> black-hat (dark screws) or top-hat (bright).
    Hat structuring element = bin short side / 12 (>> shank width).
 4. Hysteresis mask: low = Otsu of hat values on the floor, high = Otsu of
    the values above low.  Components with no 'high' pixel (faint ghost
    reflections through translucent walls) are dropped.
 5. Per component: area, skeleton (Zhang-Suen) length, endpoints, branch
    points, minAreaRect.  The single-screw unit (area, skeleton length) is
    learned from components that look like one rod (<=2 ends, elongated,
    area near the median of such rods).
 6. Per-component count = area / unit_area (exact for touching screws);
    when skeleton length / unit_length says more (overlap hides area but not
    length) the two estimates are averaged.
"""
import cv2
import numpy as np

S = 2  # analysis scale divisor


def _otsu(vals):
    vals = np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1)
    if vals.size < 2:
        return 0.0
    t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _thin(img):
    """Zhang-Suen thinning of a small binary (0/1) array."""
    P = np.pad(img.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2 = P[:-2, 1:-1]; p3 = P[:-2, 2:]; p4 = P[1:-1, 2:]; p5 = P[2:, 2:]
            p6 = P[2:, 1:-1]; p7 = P[2:, :-2]; p8 = P[1:-1, :-2]; p9 = P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(np.int32) for n in nb)
            seq = nb + [p2]
            A = sum(((seq[j] == 0) & (seq[j + 1] == 1)).astype(np.int32) for j in range(8))
            if step == 0:
                m = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            c = P[1:-1, 1:-1]
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & m
            if rm.any():
                changed = True
                c[rm] = 0
        if not changed:
            break
    return P[1:-1, 1:-1]


def _skel_stats(sk):
    """(length_px, n_endpoints, n_branch_pixels) of a 1-px skeleton."""
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    nb = cv2.filter2D(sk.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    on = sk > 0
    ends = int(((nb == 1) & on).sum())
    branches = int(((nb >= 3) & on).sum())
    s = sk.astype(np.uint8)
    orth = int((s[:, 1:] & s[:, :-1]).sum() + (s[1:, :] & s[:-1, :]).sum())
    diag = int((s[1:, 1:] & s[:-1, :-1]).sum() + (s[1:, :-1] & s[:-1, 1:]).sum())
    # diagonal links next to orthogonal ones double count; pixel count is a
    # robust length proxy with ~1px error per pixel step.
    L = float(on.sum())
    return L, ends, branches


def _segment(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    h = cv2.resize(g, (W // S, H // S), interpolation=cv2.INTER_AREA)
    q = cv2.resize(g, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    f4 = 4 // S

    # 1. bin
    _, b = cv2.threshold(cv2.GaussianBlur(q, (0, 0), 2), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(b)
    i = 1 + int(np.argmax(st[1:, 4]))
    cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    binq = np.zeros_like(q); cv2.drawContours(binq, [hull], -1, 255, -1)
    short = min(cv2.minAreaRect(hull)[1])            # bin short side, 1/4-res px
    M = cv2.moments(hull); cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

    # 2. floor. median kernel short/8 is wider than a screw (removes it)
    # but narrower than a wall band (keeps it).
    k = min(int(short / 8) | 1, 255)
    bg = cv2.medianBlur(q, k)
    t = _otsu(bg[binq > 0])
    side = (bg > t) if bg[cy, cx] > t else (bg <= t)
    side = (side & (binq > 0)).astype(np.uint8)
    _, lab2 = cv2.connectedComponents(side, connectivity=4)
    floorq = (lab2 == lab2[cy, cx]).astype(np.uint8)
    cs, _ = cv2.findContours(floorq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh = cv2.convexHull(max(cs, key=cv2.contourArea))
    floor = np.zeros_like(h); cv2.drawContours(floor, [fh * f4], -1, 255, -1)
    # erosion short/60: excludes the rim / floor-wall crease line itself
    e = max(3, int(short * f4 / 60)) | 1
    floor = cv2.erode(floor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
    fv = floor > 0

    # 3. polarity + hat
    hb = cv2.GaussianBlur(h, (0, 0), 0.8)
    coarse = cv2.resize(bg, (W // S, H // S))
    sd = hb.astype(np.int16) - coarse.astype(np.int16)
    a = np.abs(sd)
    qa = np.percentile(a[fv], 98)
    pol = float(np.sign(sd[(a >= qa) & fv]).mean())
    kh = int(short * f4 / 12) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kh, kh))
    if pol < 0:
        hat = cv2.subtract(cv2.morphologyEx(hb, cv2.MORPH_CLOSE, se), hb)
    else:
        hat = cv2.subtract(hb, cv2.morphologyEx(hb, cv2.MORPH_OPEN, se))
    hat[~fv] = 0

    # 4. hysteresis (two-level Otsu)
    v = hat[fv]
    lo = _otsu(v)
    hi = _otsu(v[v > lo])
    low = (hat > lo).astype(np.uint8)
    low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    hic = np.bincount(lab[(hat > hi) & (low > 0)], minlength=nl)
    keep = hic > 0; keep[0] = False
    mask = keep[lab].astype(np.uint8)
    # Specular highlights on polished screws have the OPPOSITE polarity
    # (bright streaks on dark screws).  Add opposite-polarity hat pixels
    # (same 'lo' threshold) but only where they lie within one hat-SE/4
    # (~ one shank width) of the main mask, so they fill screw bodies
    # without pulling in bright wall glare.
    opp = (cv2.subtract(hb, cv2.morphologyEx(hb, cv2.MORPH_OPEN, se)) if pol < 0 else
           cv2.subtract(cv2.morphologyEx(hb, cv2.MORPH_CLOSE, se), hb))
    kd = max(3, (kh // 4) | 1)
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
    mask = (mask | ((opp > lo) & (near > 0) & fv)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # opening with a disk of bin_short/50 (~10 px full-res for a 500 px bin):
    # removes structures thinner than ~2% of the bin width (moulding lines,
    # floor/wall crease, specular slivers). Screw shanks must be thicker.
    ko = max(3, int(short * f4 / 50) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    dbg = dict(open_k=ko,polarity=pol, lo=lo, hi=hi, hat_se=kh, bin_short=short * f4)
    return mask, floor, dbg


def _features(mask):
    nl, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    feats = []
    for i in range(1, nl):
        x, y, w, hh, area = (int(v) for v in st[i])
        if area < 15:   # specks: far below any head (~10x10 half-res px)
            continue
        sub = (lab[y:y + hh, x:x + w] == i).astype(np.uint8)
        L, ends, br = _skel_stats(_thin(sub))
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (_, _), (rw, rh), _ = cv2.minAreaRect(max(cs, key=len))
        feats.append(dict(area=float(area), skel=L, ends=ends, branches=br,
                          rlong=float(max(rw, rh)), rshort=float(min(rw, rh)),
                          cx=float(cen[i][0]), cy=float(cen[i][1]), box=(x, y, w, hh)))
    return feats


def _auto_unit(feats):
    rods = [f for f in feats if f['ends'] <= 2 and f['rlong'] > 2.0 * max(f['rshort'], 1)]
    if not rods:
        rods = list(feats)
    if not rods:
        return None
    a1 = np.median([f['area'] for f in rods])
    sel = [f for f in rods if a1 / 1.5 <= f['area'] <= a1 * 1.5]   # +-50%: one vs two screws
    rods = sel or rods
    return dict(area=float(np.median([f['area'] for f in rods])),
                skel=float(np.median([f['skel'] for f in rods])),
                n_rods=len(rods))


def _component_count(f, u):
    na = f['area'] / u['area']
    nl = f['skel'] / u['skel']
    n = na if nl <= na else 0.5 * (na + nl)
    if na < 0.35:          # far smaller than one screw: fragment / speck
        return 0, na, nl
    return max(1, int(round(n))), na, nl


def count_screws(frame_bgr, unit=None):
    mask, floor, dbg = _segment(frame_bgr)
    feats = _features(mask)
    u = unit if unit is not None else _auto_unit(feats)
    items, total = [], 0
    if u is not None:
        for f in feats:
            c, na, nl = _component_count(f, u)
            f.update(count=c, na=round(na, 2), nl=round(nl, 2))
            total += c
            if c > 0:
                x, y, w, hh = f['box']
                items.append(dict(box=(x * S, y * S, w * S, hh * S), n=c,
                                  xy=(f['cx'] * S, f['cy'] * S)))
    dbg.update(unit=u, comps=feats, mask=mask, floor=floor)
    return dict(count=int(total), items=items, debug=dbg)


def calibrate(frame_bgr, true_count):
    """Rescale the auto unit so this frame yields true_count."""
    mask, floor, dbg = _segment(frame_bgr)
    feats = _features(mask)
    u = _auto_unit(feats)
    if u is None or true_count <= 0:
        return u
    tot = sum(f['area'] for f in feats if f['area'] >= 0.35 * u['area'])
    r = (tot / true_count) / u['area']
    return dict(area=u['area'] * r, skel=u['skel'] * r, n_rods=u['n_rods'], calibrated=True)


def draw(frame_bgr, res):
    v = frame_bgr.copy()
    Hh, Ww = v.shape[:2]
    mm = cv2.resize(res['debug']['mask'], (Ww, Hh), interpolation=cv2.INTER_NEAREST) > 0
    v[mm] = (0.5 * v[mm] + np.array([0, 0, 127])).astype(np.uint8)
    fl = cv2.resize(res['debug']['floor'], (Ww, Hh), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(fl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v, cs, -1, (0, 255, 0), 2)
    for it in res['items']:
        x, y, w, hh = it['box']
        cv2.rectangle(v, (x, y), (x + w, y + hh), (255, 128, 0), 1)
        cv2.putText(v, str(it['n']), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(v, 'count=%d' % res['count'], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    return v
