import cv2,numpy as np,time
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    H,W=g.shape; s=4; sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA)
    _,b=cv2.threshold(cv2.GaussianBlur(sm,(0,0),2),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    nl,lab,st,_=cv2.connectedComponentsWithStats(b); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    bm=np.zeros_like(sm); cv2.drawContours(bm,[hull],-1,255,-1)
    bm=cv2.erode(bm,np.ones((5,5),np.uint8))
    md=cv2.medianBlur(sm,5)
    gm=cv2.morphologyEx(md,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    T=max(8,3*np.median(gm[bm>0]))
    flat=((gm<T)&(bm>0)).astype(np.uint8)
    bw=cv2.minAreaRect(hull)[1]; k=int(min(bw)/12)|1
    flat=cv2.morphologyEx(flat,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    nl,lab,st,_=cv2.connectedComponentsWithStats(flat,connectivity=4)
    i=1+np.argmax(st[1:,4])
    fl=(lab==i).astype(np.uint8)*255
    cs,_=cv2.findContours(fl,0,2); hull2=cv2.convexHull(max(cs,key=cv2.contourArea))
    print(n,k)
    v=cv2.cvtColor(sm,cv2.COLOR_GRAY2BGR); v[fl>0]=(v[fl>0]*0.5+np.array([0,128,0])).astype(np.uint8)
    cv2.drawContours(v,[hull2],-1,(0,0,255),1)
    cv2.imwrite('approaches/tmp/fl12_%s.png'%n,cv2.resize(v,None,fx=2,fy=2)[50:520,320:660])
