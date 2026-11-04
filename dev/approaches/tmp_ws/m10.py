import cv2,numpy as np,sys,time
sys.path.insert(0,'dev/approaches'); import watershed as W
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(f'dev/{n}.png'); t0=time.perf_counter()
    g=cv2.cvtColor(f,6); H,Wd=g.shape
    b=cv2.resize(W._bin_mask(cv2.resize(g,(Wd//2,H//2),interpolation=cv2.INTER_AREA)),(Wd,H),interpolation=0)
    x,y,w,h=cv2.boundingRect(b); gc=g[y:y+h,x:x+w].copy(); bc=b[y:y+h,x:x+w]
    gc[bc==0]=int(np.median(gc[bc>0]))
    s=4; sm=cv2.resize(gc,(w//s,h//s),interpolation=cv2.INTER_AREA)
    med=cv2.medianBlur(sm,21); dd=sm.astype(int)-med; pol=1 if np.percentile(dd,99.9)>-np.percentile(dd,0.1) else -1
    k=cv2.getStructuringElement(2,(15,15))
    bgs=cv2.morphologyEx(sm,cv2.MORPH_CLOSE if pol<0 else cv2.MORPH_OPEN,k)
    bgs=cv2.blur(bgs,(9,9))
    bg=cv2.resize(bgs,(w,h)).astype(np.float32)
    sd=(gc.astype(np.float32)-bg)*pol; sd[bc==0]=0
    pk=np.percentile(sd[bc>0],99.9); mad=np.median(np.abs(sd[bc>0]))
    print(n,'pol',pol,'pk',pk,'mad',mad,'ms',(time.perf_counter()-t0)*1000)
    v=f[y:y+h,x:x+w].copy()
    for T,c in [(0.08,(255,120,0)),(0.12,(0,200,255)),(0.4,(0,0,255))]: v[sd>T*pk]=c
    cv2.imwrite(f'dev/approaches/tmp_ws/lo_{n}.png',v)
    np.save(f'dev/approaches/tmp_ws/sd_{n}.npy',sd)
