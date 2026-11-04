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
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
    hb=cv2.GaussianBlur(h,(0,0),0.8)
    bh=cv2.subtract(cv2.morphologyEx(hb,cv2.MORPH_CLOSE,k),hb)
    th=cv2.subtract(hb,cv2.morphologyEx(hb,cv2.MORPH_OPEN,k))
    cv2.imwrite('approaches/tmp/bh_%s.png'%n,cv2.convertScaleAbs(bh,alpha=2)[25:500,320:680])
    cv2.imwrite('approaches/tmp/th_%s.png'%n,cv2.convertScaleAbs(th,alpha=2)[25:500,320:680])
