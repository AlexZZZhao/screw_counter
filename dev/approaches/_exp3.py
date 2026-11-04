import cv2,numpy as np
S=2
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); H,W=f.shape[:2]
    g=cv2.resize(cv2.cvtColor(f,6),(W//S,H//S),interpolation=cv2.INTER_AREA)
    gb=cv2.GaussianBlur(g,(0,0),2)
    _,br=cv2.threshold(gb,0,255,8)
    k=max(3,(min(g.shape)//20)|1)
    br=cv2.morphologyEx(br,cv2.MORPH_CLOSE,cv2.getStructuringElement(2,(k,k)))
    nl,lab,st,_=cv2.connectedComponentsWithStats(br); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    binm=np.zeros_like(g); cv2.drawContours(binm,[hull],-1,255,-1)
    bw=st[i,2]; e=(bw//40)|1; binm=cv2.erode(binm,cv2.getStructuringElement(2,(e,e)))
    sm=cv2.resize(g,(W//8,H//8),interpolation=cv2.INTER_AREA)
    bg=cv2.resize(cv2.medianBlur(sm,15),(W//S,H//S)).astype(np.float32)
    d=np.abs(g.astype(np.float32)-bg)
    v=d[binm>0]
    t,_=cv2.threshold(np.clip(v,0,255).astype(np.uint8).reshape(-1,1),0,255,8)
    print(n,'otsu',t,'pct',np.percentile(v,[50,90,95,98,99,99.5]))
    m=((d>t)&(binm>0)).astype(np.uint8)*255
    strong=((d>2*t)&(binm>0)).astype(np.uint8)*255
    vis=cv2.cvtColor(g,8); vis[m>0]=(0,200,255); vis[strong>0]=(0,0,255)
    cv2.imwrite('approaches/_m2_'+n+'.png',vis)
