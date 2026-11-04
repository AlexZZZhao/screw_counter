import cv2,numpy as np
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    H,W=g.shape; s=4
    sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA)
    bg=cv2.medianBlur(sm,31)
    hist=np.bincount(bg.ravel(),minlength=256)
    print(n, 'center bg', bg[H//s//2-20:H//s//2+20:10, W//s//2-20:W//s//2+20:10].ravel())
    cv2.imwrite('approaches/tmp/bg_'+n+'.png',bg)
