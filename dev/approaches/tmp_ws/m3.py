import cv2,numpy as np,sys,time
sys.path.insert(0,'dev/approaches/tmp_ws'); from m2 import binmask
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(f'dev/{n}.png'); g=cv2.cvtColor(f,6); H,W=g.shape
    b=binmask(g); b=cv2.erode(b,cv2.getStructuringElement(2,(31,31)))
    s=4; bg=cv2.resize(cv2.medianBlur(cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA),21),(W,H)).astype(np.float32)
    d=cv2.GaussianBlur(g,(0,0),1).astype(np.float32)-bg; d[b==0]=0
    pos=np.percentile(d[b>0],99.9); neg=-np.percentile(d[b>0],0.1); pol=1 if pos>neg else -1
    sd=d*pol; peak=max(pos,neg); print(n,'pol',pol,'peak',peak)
    v=f.copy()
    for k,c in [(0.5,(0,0,255)),(0.35,(0,200,255)),(0.2,(255,0,0))]:
        pass
    v[(sd>0.2*peak)]=(255,0,0); v[(sd>0.35*peak)]=(0,200,255); v[(sd>0.5*peak)]=(0,0,255)
    cv2.imwrite(f'dev/approaches/tmp_ws/t_{n}.png',v[100:1000,650:1300])
