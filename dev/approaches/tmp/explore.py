import cv2,numpy as np,sys
sys.path.insert(0,'dev'); from proto3 import run
for n in ['gt27_0','frame_focused']:
    f=cv2.imread('dev/'+n+'.png'); m,b=run(f)
    dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
    sk=cv2.ximgproc.thinning(m) if hasattr(cv2,'ximgproc') else None
    # ridge: local max of dt in 3x3
    ridge=(dt>=cv2.dilate(dt,np.ones((3,3))))&(dt>1)
    h=np.bincount(np.round(dt[ridge]).astype(int))
    print(n,'ridge dt hist',list(h[:30]))
    for r in (6,8,10,12,14,18,22):
        o=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
        nn=cv2.connectedComponents(o)[0]-1
        print(' open r',r,'comps',nn)
