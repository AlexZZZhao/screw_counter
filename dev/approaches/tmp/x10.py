import cv2,numpy as np,time
def prep(g):
    H,W=g.shape; s=4; sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA)
    _,b=cv2.threshold(cv2.GaussianBlur(sm,(0,0),2),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    nl,lab,st,_=cv2.connectedComponentsWithStats(b); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    bm=np.zeros((H//2,W//2),np.uint8); cv2.drawContours(bm,[hull*2],-1,255,-1)
    return cv2.erode(bm,np.ones((9,9),np.uint8))
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    H,W=g.shape; h=cv2.resize(g,(W//2,H//2),interpolation=cv2.INTER_AREA)
    bm=prep(g)
    s8=cv2.resize(g,(W//8,H//8),interpolation=cv2.INTER_AREA)
    bg=cv2.resize(cv2.medianBlur(s8,13),(W//2,H//2))
    sd=h.astype(np.int16)-bg; a=np.abs(sd); q=np.percentile(a[bm>0],98)
    pol=np.sign(sd[(a>=q)&(bm>0)]).mean()
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)); hb=cv2.GaussianBlur(h,(0,0),0.8)
    hat=cv2.subtract(cv2.morphologyEx(hb,cv2.MORPH_CLOSE,k),hb) if pol<0 else cv2.subtract(hb,cv2.morphologyEx(hb,cv2.MORPH_OPEN,k))
    hat[bm==0]=0
    v=hat[bm>0]; lo,_=cv2.threshold(v.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    hi,_=cv2.threshold(v[v>lo].reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    low=(hat>lo).astype(np.uint8)
    nl,lab,st,_=cv2.connectedComponentsWithStats(low,connectivity=8)
    hic=np.bincount(lab[hat>hi],minlength=nl)
    keep=hic>=1; keep[0]=False
    m=keep[lab].astype(np.uint8)*255
    dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
    nl,lab,st,_=cv2.connectedComponentsWithStats(m)
    mx=np.zeros(nl); np.maximum.at(mx,lab.ravel(),dt.ravel())
    print(n,lo,hi,'comp dtmax',np.round(np.sort(mx[1:])[::-1][:60],1))
    v=cv2.cvtColor(h,cv2.COLOR_GRAY2BGR); v[m>0]=(0,0,255)
    cv2.imwrite('approaches/tmp/m10_%s.png'%n,v[25:500,320:680])
