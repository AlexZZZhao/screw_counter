import cv2,numpy as np,sys,time
sys.path.insert(0,'dev/approaches/tmp'); from common import contrast
def lines(L,nang=16):
    ks=[]
    for k in range(nang):
        a=np.pi*k/nang; K=np.zeros((L,L),np.uint8); c=(L-1)/2
        dx,dy=np.cos(a)*c,np.sin(a)*c
        cv2.line(K,(int(round(c-dx)),int(round(c-dy))),(int(round(c+dx)),int(round(c+dy))),1,1)
        ks.append(K)
    return ks
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread('dev/'+n+'.png'); g,d,bm=contrast(f)
    v=d[bm>0]; t1,_=cv2.threshold(v,0,255,8); t2,_=cv2.threshold(v[v>t1],0,255,8)
    t=(t1+t2)/2
    m=((d>t)*255).astype(np.uint8); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(2,(5,5)))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
    ridge=(dt>=cv2.dilate(dt,np.ones((3,3))))&(dt>1.5)
    h=np.bincount(np.round(dt[ridge]).astype(int),weights=d[ridge].astype(float)); r=int(np.argmax(h[2:]))+2
    w=2*r; L=int(round(2.5*w))|1
    ys,xs=np.nonzero(m); x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    mr=m[y0:y1,x0:x1]
    t0=time.time()
    U=np.zeros_like(mr)
    for K in lines(L): U|=cv2.morphologyEx(mr,cv2.MORPH_OPEN,K)
    res=cv2.subtract(mr,U)
    print(n,'t',t,'w',w,'L',L,'ms',(time.time()-t0)*1000, mr.shape)
    vis=f[y0:y1,x0:x1].copy(); vis[mr>0]=(vis[mr>0]*0.5+np.array([0,0,100])).astype(np.uint8); vis[res>0]=(0,255,0)
    cv2.imwrite('dev/approaches/tmp/lo_'+n+'.png',vis)
