import cv2,numpy as np,time
for n in ['gt27_0','gt27_2','frame_focused','white']:
    f=cv2.imread(n+'.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    t0=time.time()
    H,W=g.shape; s=4; sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA)
    _,b=cv2.threshold(cv2.GaussianBlur(sm,(0,0),2),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    nl,lab,st,_=cv2.connectedComponentsWithStats(b); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    bm=np.zeros_like(sm); cv2.drawContours(bm,[hull],-1,255,-1)
    bg=cv2.medianBlur(sm,9)
    gx=cv2.Sobel(bg,cv2.CV_32F,1,0); gy=cv2.Sobel(bg,cv2.CV_32F,0,1); gm=cv2.magnitude(gx,gy)
    v=gm[bm>0]
    print(n,np.percentile(v,[50,75,90,95]))
    cv2.imwrite('approaches/tmp/gm_%s.png'%n,cv2.convertScaleAbs(gm,alpha=0.5))
