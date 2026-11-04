import cv2,numpy as np,sys
sys.path.insert(0,'dev/approaches/tmp'); from common import contrast
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread('dev/'+n+'.png'); g,d,bm=contrast(f)
    v=d[bm>0]; print(n,'pct',np.percentile(v,[90,95,98,99,99.5]))
    for t in (30,45,60):
        m=(d>t).astype(np.uint8)*255; m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(2,(5,5)))
        dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
        ridge=(dt>=cv2.dilate(dt,np.ones((3,3))))&(dt>1.5)
        # weight by contrast
        w=d[ridge].astype(float); r=np.round(dt[ridge]).astype(int)
        h=np.bincount(r,weights=w,minlength=25)[:25]
        print(' t',t,'weighted ridge radius hist',(h/h.max()*100).astype(int))
