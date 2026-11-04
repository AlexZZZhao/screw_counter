import os, sys, time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import segments as S


def load(n):
    return cv2.imread(os.path.join(DEV, n + '.png'))


def overlay(f, res, path):
    v = f.copy()
    floor = cv2.resize(res['debug']['floor'], f.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v, cs, -1, (255, 255, 0), 2)
    cols = {1: (0, 200, 0), 2: (0, 140, 255), 3: (255, 0, 255), 4: (0, 0, 255)}
    for i, it in enumerate(res['items']):
        p0 = tuple(int(c) for c in it['p0']); p1 = tuple(int(c) for c in it['p1'])
        c = cols.get(it['n'], (0, 0, 255))
        cv2.line(v, p0, p1, c, 3)
        cv2.circle(v, p0, 5, c, -1); cv2.circle(v, p1, 5, c, -1)
        m = ((p0[0] + p1[0]) // 2 + 6, (p0[1] + p1[1]) // 2)
        cv2.putText(v, str(i) + ('x%d' % it['n'] if it['n'] > 1 else ''), m,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(v, 'count=%d' % res['count'], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 4)
    cv2.imwrite(path, v)


def timed(f, unit=None, reps=5):
    S.count_screws(f, unit)  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); r = S.count_screws(f, unit); ts.append(time.perf_counter() - t)
    return r, 1000 * np.mean(ts)


names = ['gt27_0', 'gt27_1', 'gt27_2', 'gt27_3', 'gt27_4', 'frame_focused', 'white']
truth = dict(gt27_0=27, gt27_1=27, gt27_2=27, gt27_3=27, gt27_4=27, frame_focused=19)
frames = {n: load(n) for n in names}
allms = []
print('== uncalibrated (unit=None) ==')
for n in names:
    r, ms = timed(frames[n]); allms.append(ms)
    d = r['debug']
    print('%-14s count=%3d truth=%-4s W=%.1f L=%.1f rods=%d  %.1f ms' %
          (n, r['count'], truth.get(n, '?'), d['W'], d['L'], len(d['lens']), ms))
    if n == 'gt27_0':
        overlay(frames[n], r, os.path.join(HERE, 'segments_gt27_0.png'))
    if n == 'frame_focused':
        overlay(frames[n], r, os.path.join(HERE, 'segments_focused.png'))
print('mean ms: %.1f' % np.mean(allms))

print('== calibrated on gt27_0 (27) ==')
u = S.calibrate(frames['gt27_0'], 27)
print('unit', u)
for n in names[1:5] + ['white']:
    r = S.count_screws(frames[n], u)
    print('%-14s count=%3d truth=%s' % (n, r['count'], truth.get(n, '?')))
print('== calibrated on frame_focused (19) ==')
u2 = S.calibrate(frames['frame_focused'], 19)
r = S.count_screws(frames['frame_focused'], u2)
print('unit', u2, 'count', r['count'])
