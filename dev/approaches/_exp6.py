import cv2,numpy as np
S=2
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); H,W=f.shape[:2]
    gf=cv2.cvtColor(f,6)
    g=cv2.resize(gf,(W//S,H//S),interpolation=cv2.INTER_AREA).astype(np.float32)
    sm=cv2.resize(gf,(W//8,H//8),interpolation=cv2.INTER_AREA)
    _,br=cv2.threshold(cv2.GaussianBlur(sm,(0,0),1),0,255,8)
    nl,lab,st,cen=cv2.connectedComponentsWithStats(br); i=1+np.argmax(st[1:,4])
    x,y,bw,bh=st[i,:4]
    med=cv2.medianBlur(sm,9).astype(np.float32); d=sm-med
    cb=np.zeros_like(sm); cb[y+bh//4:y+3*bh//4,x+bw//4:x+3*bw//4]=1
    pol=np.sign(d[(np.abs(d)>20)&(cb>0)].sum())
    k=cv2.getStructuringElement(2,(7,7))
    clean=cv2.medianBlur(cv2.morphologyEx(sm,cv2.MORPH_CLOSE if pol<0 else cv2.MORPH_OPEN,k),5)
    ff=np.zeros((sm.shape[0]+2,sm.shape[1]+2),np.uint8)
    cv2.floodFill(clean.copy(),ff,(int(cen[i][0]),int(cen[i][1])),0,2,2,4|cv2.FLOODFILL_MASK_ONLY|(255<<8))
    fl=(ff[1:-1,1:-1]>0)&(lab==i)
    # floor samples: flood & not object
    samp=(fl&(np.abs(d)<15)).astype(np.float32)
    sig=bw/8
    num=cv2.GaussianBlur(sm.astype(np.float32)*samp,(0,0),sig); den=cv2.GaussianBlur(samp,(0,0),sig)
    bg=num/np.maximum(den,1e-3)
    bgS=cv2.resize(bg,(W//S,H//S))
    c=(g-bgS)*pol
    cs,_=cv2.findContours(fl.astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))*(8//S)
    floor=np.zeros(g.shape,np.uint8); cv2.drawContours(floor,[hull],-1,255,-1)
    v=c[floor>0]; t,_=cv2.threshold(np.clip(v,0,255).astype(np.uint8).reshape(-1,1),0,255,8)
    print(n,'otsu',t,np.percentile(v,[50,95,99]))
    vis=np.clip(c*2,0,255).astype(np.uint8); vis=cv2.cvtColor(vis,8)
    cv2.drawContours(vis,[hull],-1,(0,255,0),1)
    cv2.imwrite('approaches/_c_'+n+'.png',vis)
