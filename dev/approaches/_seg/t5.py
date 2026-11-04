import cv2,numpy as np,sys,time
sys.path.insert(0,'dev'); from proto3 import run
for n,crop in [('gt27_0',(280,800,740,1220)),('frame_focused',(580,840,880,1180))]:
    f=cv2.imread('dev/'+n+'.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    t=time.time(); l=lsd.detect(g)[0].reshape(-1,4); print(n,len(l),time.time()-t)
    gh=cv2.resize(g,None,fx=.5,fy=.5); t=time.time(); l2=lsd.detect(gh)[0]; print('half',len(l2),time.time()-t)
    v=f.copy()
    for (x1,y1,x2,y2) in l.reshape(-1,4):
        L=np.hypot(x2-x1,y2-y1)
        if L<15: continue
        # color by direction (polarity)
        a=np.degrees(np.arctan2(y2-y1,x2-x1))%360
        c=tuple(int(c) for c in cv2.cvtColor(np.uint8([[[a/2,255,255]]]),cv2.COLOR_HSV2BGR)[0,0])
        cv2.line(v,(int(x1),int(y1)),(int(x2),int(y2)),c,1)
    y0,y1_,x0,x1_=crop; cv2.imwrite('dev/approaches/_seg/l_'+n+'.png',cv2.resize(v[y0:y1_,x0:x1_],None,fx=2,fy=2))
