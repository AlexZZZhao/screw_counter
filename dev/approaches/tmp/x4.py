import cv2,numpy as np,time
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); t0=time.time(); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    H,W=g.shape
    h=cv2.resize(g,(W//2,H//2),interpolation=cv2.INTER_AREA)
    s8=cv2.resize(g,(W//8,H//8),interpolation=cv2.INTER_AREA)
    bg=cv2.resize(cv2.medianBlur(s8,13),(W//2,H//2),interpolation=cv2.INTER_LINEAR)
    d=cv2.absdiff(cv2.GaussianBlur(h,(0,0),1),bg)
    print(n,time.time()-t0)
    cv2.imwrite('approaches/tmp/d_'+n+'.png',cv2.convertScaleAbs(d,alpha=2)[50:500,320:680])
