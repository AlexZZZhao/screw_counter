import cv2,numpy as np,sys
sys.path.insert(0,'dev/approaches'); import watershed as W
for n in ['gt27_0','frame_focused']:
    f=cv2.imread(f'dev/{n}.png'); g=cv2.cvtColor(f,6); H,Wd=g.shape
    b=cv2.resize(W._bin_mask(cv2.resize(g,(Wd//2,H//2))),(Wd,H),interpolation=0)
    gf=g.copy(); gf[b==0]=int(np.median(g[b>0]))
    s=8; bg=cv2.resize(cv2.medianBlur(cv2.resize(gf,(Wd//s,H//s),interpolation=cv2.INTER_AREA),21),(Wd,H)).astype(np.float32)
    d=gf.astype(np.float32)-bg
    pos,neg=np.percentile(d[b>0],99.9),-np.percentile(d[b>0],0.1); pol=1 if pos>neg else -1; pk=max(pos,neg); sd=d*pol
    v=f.copy(); v[sd>0.4*pk]=(0,0,255); v[(sd>0.15*pk)&(sd<=0.4*pk)]=(255,120,0); v[-sd>0.15*pk]=(0,220,0)
    cv2.imwrite(f'dev/approaches/tmp_ws/mk_{n}.png',v)
    print(n,pk)
