import cv2,numpy as np
def prep(n):
    f=cv2.imread('dev/'+n+'.png'); g=cv2.cvtColor(f,6); H,W=g.shape; s=8
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
    M=cv2.moments(bms); cx,cy=int(M['m10']/M['m00']),int(M['m01']/M['m00'])
    out=[]
    for tol in (2,3,4,6):
        ff=np.zeros((sm.shape[0]+2,sm.shape[1]+2),np.uint8)
        bgb=cv2.GaussianBlur(bg,(0,0),0.7)
        cv2.floodFill(bgb.copy(),ff,(cx,cy),0,tol,tol,4|cv2.FLOODFILL_MASK_ONLY|(255<<8))
        fl=ff[1:-1,1:-1]&bms
        v=cv2.cvtColor(sm,8); v[fl>0]=(v[fl>0]*0.5+np.array([0,128,0])).astype(np.uint8)
        out.append(cv2.resize(v,None,fx=2,fy=2,interpolation=0)[10:260,150:330])
    cv2.imwrite('dev/approaches/tmp/fl_'+n+'.png',np.hstack(out))
for n in ['gt27_0','frame_focused','white']: prep(n)
