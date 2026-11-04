import cv2,numpy as np,sys
sys.path.insert(0,'dev'); from proto3 import run
for n in ['gt27_0','frame_focused']:
    f=cv2.imread('dev/'+n+'.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    H,W=g.shape; s=4; bg=cv2.resize(cv2.medianBlur(cv2.resize(g,(W//s,H//s)),15),(W,H))
    d=cv2.absdiff(g,bg)
    cv2.imwrite('dev/approaches/_seg/d_'+n+'.png',cv2.resize(np.clip(d.astype(int)*3,0,255).astype(np.uint8),(960,540)))
    print(n,np.percentile(d,[50,90,99]))
