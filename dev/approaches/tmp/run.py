import sys, cv2, time
sys.path.insert(0, '.')
import skeleton as S
for n in sys.argv[1:]:
    f = cv2.imread('../%s.png' % n); t = time.time(); r = S.count_screws(f); dt = time.time() - t
    d = r['debug']
    print(n, r['count'], '%.0fms' % (dt * 1000), 'pol%.2f lo%.0f hi%.0f' % (d['polarity'], d['lo'], d['hi']), d['unit'])
    for c in d['comps']:
        print('  ', c['box'], c['area'], round(c['skel']), c['ends'], c['branches'], c.get('count'), c.get('na'), c.get('nl'))
    v = S.draw(f, r)
    cv2.imwrite('tmp/o_%s.png' % n, v[50:1000, 640:1300])
