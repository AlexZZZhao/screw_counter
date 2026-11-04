import cv2,numpy as np
S=2
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); H,W=f.shape[:2]
    gf=cv2.cvtColor(f,6)
    g=cv2.resize(gf,(W//S,H//S),interpolation=cv2.INTER_AREA)
    sm=cv2.resize(gf,(W//8,H//8),interpolation=cv2.INTER_AREA)
    _,br=cv2.threshold(cv2.GaussianBlur(sm,(0,0),1),0,255,8)
    nl,lab,st,cen=cv2.connectedComponentsWithStats(br); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    binS=np.zeros_like(sm); cv2.drawContours(binS,[hull],-1,255,-1)
    x,y,bw,bh=st[i,:4]
    med=cv2.medianBlur(sm,9).astype(np.float32)
    d=sm.astype(np.float32)-med
    cb=np.zeros_like(sm); cb[y+bh//4:y+3*bh//4,x+bw//4:x+3*bw//4]=1
    sel=(np.abs(d)>20)&(cb>0)
    pol=np.sign(d[sel].sum())
    k=cv2.getStructuringElement(2,(7,7))
    clean=cv2.morphologyEx(sm,cv2.MORPH_CLOSE if pol<0 else cv2.MORPH_OPEN,k)
    clean=cv2.medianBlur(clean,5)
    ff=np.zeros((sm.shape[0]+2,sm.shape[1]+2),np.uint8)
    seed=(int(cen[i][0]),int(cen[i][1]))
    cv2.floodFill(clean.copy(),ff,seed,0,2,2,4|cv2.FLOODFILL_MASK_ONLY|(255<<8))
    fl=ff[1:-1,1:-1]&binS
    print(n,'pol',pol,'flood frac',(fl>0).sum()/(binS>0).sum())
    v=cv2.cvtColor(clean,8); v[fl>0]=(v[fl>0]*0.5+[0,100,0]).astype(np.uint8)
    cv2.imwrite('approaches/_fl_'+n+'.png',cv2.resize(v,(960,540),interpolation=cv2.INTER_NEAREST))
