import cv2,numpy as np,sys
sys.path.insert(0,'dev/approaches/tmp_ws'); from m2 import binmask
for n in ['gt27_0','frame_focused']:
    f=cv2.imread(f'dev/{n}.png'); g=cv2.cvtColor(f,6); H,W=g.shape
    b=cv2.erode(binmask(g),cv2.getStructuringElement(2,(31,31)))
    s=4; bg=cv2.resize(cv2.medianBlur(cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA),21),(W,H)).astype(np.float32)
    d=cv2.GaussianBlur(g,(0,0),1).astype(np.float32)-bg; d[b==0]=0
    pos=np.percentile(d[b>0],99.9); neg=-np.percentile(d[b>0],0.1); pol=1 if pos>neg else -1
    sd=d*pol; peak=max(pos,neg)
    m=((sd>0.4*peak)*255).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(2,(5,5)))
    dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
    nl,lab,st,cen=cv2.connectedComponentsWithStats(m)
    v=f.copy(); v[m>0]=(0,0,255)
    for i in range(1,nl):
        if st[i,4]<100: continue
        cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); c=max(cs,key=cv2.contourArea)
        (cx,cy),(w,h),a=cv2.minAreaRect(c); L=max(w,h); dm=dt[lab==i].max()
        print(n,i,st[i,4],'L',round(L),'W',round(min(w,h)),'dtmax',round(dm,1),'fill',round(st[i,4]/(w*h+1),2))
        cv2.putText(v,str(i),(int(cx),int(cy)),0,0.8,(0,255,0),2)
    cv2.imwrite(f'dev/approaches/tmp_ws/cc_{n}.png',v[100:1000,650:1300])
