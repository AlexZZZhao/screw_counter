import cv2,sys,time
sys.path.insert(0,'dev/approaches'); import watershed as W
for n in ['gt27_0','gt27_1','gt27_2','gt27_3','gt27_4','frame_focused','white']:
    f=cv2.imread(f'dev/{n}.png'); t=time.perf_counter(); r=W.count_screws(f); dt=(time.perf_counter()-t)*1000
    d=r['debug']; print(n,r['count'],f"{dt:.0f}ms ua={d['unit_area']:.0f} hw={d['half_w']:.1f} thick={d['thick']:.1f} inst={d['n_inst']} cand={d['n_cand']}")
    cv2.imwrite(f'dev/approaches/tmp_ws/o_{n}.png',W.overlay(f,r)[100:1000,650:1300])
