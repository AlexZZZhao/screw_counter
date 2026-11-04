import cv2,numpy as np
for n in ['gt27_0','gt27_3','frame_focused','white']:
    f=cv2.imread(n+'.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    H,W=g.shape; s=4
    sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA)
    _,b=cv2.threshold(cv2.GaussianBlur(sm,(0,0),2),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    nl,lab,st,_=cv2.connectedComponentsWithStats(b); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    binm=np.zeros_like(sm); cv2.drawContours(binm,[hull],-1,255,-1)
    # local background via closing/opening polarity-agnostic: median
    bg=cv2.medianBlur(sm,15)
    vals=bg[binm>0]
    t,_=cv2.threshold(vals.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fl=((bg<=t)&(binm>0)).astype(np.uint8)*255
    print(n,t, np.percentile(vals,[5,25,50,75,95]))
    v=cv2.cvtColor(sm,cv2.COLOR_GRAY2BGR); v[fl>0]=(v[fl>0]*0.5+np.array([0,128,0])).astype(np.uint8)
    cv2.imwrite('approaches/tmp/fl_'+n+'.png',v)
