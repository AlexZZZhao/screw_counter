import cv2,numpy as np,sys
sys.path.insert(0,'dev/approaches/tmp'); from common import contrast
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread('dev/'+n+'.png'); g,d,bm=contrast(f)
    v=d[bm>0]; t1,_=cv2.threshold(v,0,255,8); t2,_=cv2.threshold(v[v>t1],0,255,8)
    print(n,t1,t2)
    m=((d>t2)*255).astype(np.uint8); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(2,(5,5)))
    dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
    pk=(dt>=cv2.dilate(dt,cv2.getStructuringElement(2,(9,9))))&(dt>=5)
    ys,xs=np.nonzero(pk)
    vis=cv2.cvtColor(np.clip(dt*15,0,255).astype(np.uint8),8)
    vis=cv2.addWeighted(vis,0.6,f,0.4,0)
    for x,y in zip(xs,ys):
        cv2.putText(vis,str(int(dt[y,x])),(x-6,y+4),0,0.4,(0,0,255),1)
    cv2.imwrite('dev/approaches/tmp/pk_'+n+'.png',vis[250:900,700:1250] if n!='white' else vis[450:850,700:1250])
