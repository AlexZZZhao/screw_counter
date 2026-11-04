import cv2,numpy as np
def contrast(f):
    g=cv2.cvtColor(f,6); H,W=g.shape; s=8
    sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA)
    _,b=cv2.threshold(cv2.GaussianBlur(sm,(0,0),1),0,255,8)
    nn,lab,st,_=cv2.connectedComponentsWithStats(b); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    bms=np.zeros_like(sm); cv2.drawContours(bms,[hull],-1,255,-1)
    bg=cv2.medianBlur(sm,9)
    for it in range(2):
        d=cv2.absdiff(sm,bg); fg=((d>12)&(bms>0)).astype(np.uint8)*255
        fg=cv2.dilate(fg,np.ones((3,3)))|(bms==0).astype(np.uint8)*255
        bg=cv2.inpaint(sm,fg,5,cv2.INPAINT_TELEA)
    BG=cv2.resize(bg,(W,H)); bm=cv2.resize(bms,(W,H),interpolation=0)
    d=cv2.absdiff(cv2.GaussianBlur(g,(0,0),1.2),BG); d[cv2.erode(bm,np.ones((15,15)))==0]=0
    return g,d,bm
