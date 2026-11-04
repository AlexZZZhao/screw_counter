import cv2,numpy as np,sys
sys.path.insert(0,'dev'); from proto3 import run
for n in ['gt27_0','frame_focused','white','gt27_3']:
    f=cv2.imread('dev/'+n+'.png'); m,b=run(f)
    s=4; g=cv2.cvtColor(cv2.resize(f,None,fx=1/s,fy=1/s,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    bs=cv2.resize(b,g.shape[::-1],interpolation=cv2.INTER_NEAREST)
    bg=cv2.medianBlur(cv2.medianBlur(g,9),9)
    M=cv2.moments(bs); cx,cy=int(M['m10']/M['m00']),int(M['m01']/M['m00'])
    for tol in (2,):
        mk=np.zeros((g.shape[0]+2,g.shape[1]+2),np.uint8)
        im=bg.copy(); im[bs==0]=0
        cv2.floodFill(im,mk,(cx,cy),255,tol,tol,4|cv2.FLOODFILL_MASK_ONLY|(255<<8))
        fl=mk[1:-1,1:-1]
        print(n,tol,fl.sum()/255/ (bs>0).sum())
    v=cv2.resize(f,g.shape[::-1]); v[fl>0]=(v[fl>0]*.5+np.array([0,128,0])).astype(np.uint8)
    cv2.imwrite('dev/approaches/_seg/ff_'+n+'.png',cv2.resize(v,None,fx=2,fy=2))
