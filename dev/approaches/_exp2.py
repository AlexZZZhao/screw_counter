import cv2,numpy as np
def fg(f):
    s=2; g=cv2.cvtColor(f,6); H,W=g.shape
    g=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA).astype(np.float32)
    small=cv2.resize(g,(W//8,H//8),interpolation=cv2.INTER_AREA).astype(np.uint8)
    bg=cv2.resize(cv2.medianBlur(small,15).astype(np.float32),(W//s,H//s))
    d=g-bg
    return g,bg,d
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); g,bg,d=fg(f)
    a=np.abs(d); print(n,'MAD',np.median(a),'pct',np.percentile(a,[90,99,99.5]),'neg>40',(d<-40).sum(),'pos>40',(d>40).sum())
    v=np.clip(128+d*2,0,255).astype(np.uint8); cv2.imwrite('approaches/_d_'+n+'.png',v)
