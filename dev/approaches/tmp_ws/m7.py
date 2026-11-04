import cv2,numpy as np
f=cv2.imread('dev/gt27_0.png'); g=cv2.cvtColor(f,6); H,W=g.shape
s=4; bg=cv2.resize(cv2.medianBlur(cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA),21),(W,H)).astype(np.float32)
d=g.astype(np.float32)-bg
for y in (320,350,400): print(y,' '.join(f'{int(v)}' for v in d[y,805:875]))
for y in (520,560,600): print(y,' '.join(f'{int(v)}' for v in d[y,1125:1185]))
