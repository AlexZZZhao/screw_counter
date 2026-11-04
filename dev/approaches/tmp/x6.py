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
    H,W=g.shape; h=cv2.resize(g,(W//2,H//2),interpolation=cv2.INTER_AREA)
    s8=cv2.resize(g,(W//8,H//8),interpolation=cv2.INTER_AREA)
    bg=cv2.resize(cv2.medianBlur(s8,13),(W//2,H//2))
    d=cv2.absdiff(cv2.GaussianBlur(h,(0,0),0.8),bg)
    bm=cv2.resize(binmask(g),(W//2,H//2),interpolation=cv2.INTER_NEAREST)
    bm=cv2.erode(bm,np.ones((9,9),np.uint8))
    v=d[bm>0]
    t,_=cv2.threshold(v.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(n,'otsu',t,np.percentile(v,[50,90,95,98,99]))
    hi=v[v>t]; t2,_=cv2.threshold(hi.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print('  second otsu',t2)
    for tt in (t,t2,(t+t2)/2):
        m=((d>tt)&(bm>0)).astype(np.uint8)*255
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        cv2.imwrite('approaches/tmp/m_%s_%d.png'%(n,tt),m[25:500,320:680])
