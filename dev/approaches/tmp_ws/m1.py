import cv2,numpy as np,sys
sys.path.insert(0,'dev')
from proto3 import run
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(f'dev/{n}.png'); _,b=run(f)
    g=cv2.cvtColor(f,6); H,W=g.shape; s=4
    bg=cv2.resize(cv2.medianBlur(cv2.resize(g,(W//s,H//s)),15),(W,H)).astype(np.float32)
    d=g.astype(np.float32)-bg
    dd=d[b>0]; print(n,'pos',np.percentile(dd,[99,99.5,99.9]).round(),'neg',np.percentile(dd,[0.1,0.5,1]).round(), 'mad',np.median(np.abs(dd)))
    v=np.clip(128+d,0,255).astype(np.uint8); v[b==0]=0
    cv2.imwrite(f'dev/approaches/tmp_ws/d_{n}.png',v[100:1000,650:1300])
