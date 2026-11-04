import cv2,numpy as np
def binmask(g):
    H,W=g.shape; s=8; sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA)
    _,b=cv2.threshold(cv2.GaussianBlur(sm,(0,0),1),0,255,8)
    n,lab,st,_=cv2.connectedComponentsWithStats(b)
    i=1+np.argmax(st[1:,4]); cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2)
    hull=cv2.convexHull(max(cs,key=cv2.contourArea)); bm=np.zeros_like(sm); cv2.drawContours(bm,[hull],-1,255,-1)
    return cv2.resize(bm,(W,H),interpolation=0), sm, s
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread('dev/'+n+'.png'); g=cv2.cvtColor(f,6); H,W=g.shape
    bm,sm,s=binmask(g)
    bms=cv2.resize(bm,sm.shape[::-1],interpolation=0)
    bg=cv2.medianBlur(sm,9)
    for it in range(2):
        d=cv2.absdiff(sm,bg); fg=((d>12)&(bms>0)).astype(np.uint8)*255
        fg=cv2.dilate(fg,np.ones((3,3)))|(bms==0).astype(np.uint8)*255
        bg=cv2.inpaint(sm,fg,5,cv2.INPAINT_TELEA)
    BG=cv2.resize(bg,(W,H),interpolation=cv2.INTER_LINEAR)
    d=cv2.absdiff(cv2.GaussianBlur(g,(0,0),1.2),BG); d[bm==0]=0
    din=d[cv2.erode(bm,np.ones((41,41)))>0]
    print(n,np.percentile(din,[50,75,90,99,99.9]),np.median(np.abs(din-np.median(din))))
    v=cv2.applyColorMap(np.clip(d*2,0,255).astype(np.uint8),cv2.COLORMAP_JET)
    cv2.imwrite('dev/approaches/tmp/d2_'+n+'.png',np.hstack([v[250:900,700:1250],cv2.cvtColor(BG,8)[250:900,700:1250]]))
