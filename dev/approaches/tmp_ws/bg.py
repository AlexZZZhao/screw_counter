import cv2,numpy as np,sys
for n in ['gt27_0','frame_focused']:
    f=cv2.imread(f'dev/{n}.png'); g=cv2.cvtColor(f,6)
    s=cv2.resize(g,(192,108),interpolation=cv2.INTER_AREA)
    s=cv2.medianBlur(s,5)
    print(n)
    for y in range(10,108,4): print(' '.join(f'{v:3d}' for v in s[y,66:132:3]))
