import cv2,numpy as np,time
def binmask(g):
    H,W=g.shape; sm=cv2.resize(g,(W//4,H//4),interpolation=cv2.INTER_AREA)
    _,b=cv2.threshold(cv2.GaussianBlur(sm,(0,0),2),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    nl,lab,st,_=cv2.connectedComponentsWithStats(b); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    m=np.zeros_like(sm); cv2.drawContours(m,[hull],-1,255,-1)
    return m
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    t0=time.time()
    H,W=g.shape; h=cv2.resize(g,(W//2,H//2),interpolation=cv2.INTER_AREA)
    bm=cv2.resize(binmask(g),(W//2,H//2),interpolation=cv2.INTER_NEAREST)
    bm=cv2.erode(bm,np.ones((9,9),np.uint8))
    s8=cv2.resize(g,(W//8,H//8),interpolation=cv2.INTER_AREA)
    bg=cv2.resize(cv2.medianBlur(s8,13),(W//2,H//2))
    sd=h.astype(np.int16)-bg
    a=np.abs(sd)[bm>0]; q=np.percentile(a,98)
    sel=(np.abs(sd)>=q)&(bm>0)
    pol=np.sign(sd[sel]).mean()
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)); hb=cv2.GaussianBlur(h,(0,0),0.8)
    hat=cv2.subtract(cv2.morphologyEx(hb,cv2.MORPH_CLOSE,k),hb) if pol<0 else cv2.subtract(hb,cv2.morphologyEx(hb,cv2.MORPH_OPEN,k))
    v=hat[bm>0]; t,_=cv2.threshold(v.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    m=((hat>t)&(bm>0)).astype(np.uint8)*255
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    dt=cv2.distanceTransform(m,cv2.DIST_L2,3)
    # width estimate: local maxima of dt
    lm=(dt>=cv2.dilate(dt,np.ones((3,3))))&(dt>1)
    print(n,'pol',pol,'t',t,'dt ridge pct',np.percentile(dt[lm],[25,50,75,90]),time.time()-t0)
    cv2.imwrite('approaches/tmp/m8_%s.png'%n,m[25:500,320:680])
