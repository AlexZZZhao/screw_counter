import cv2,numpy as np
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(n+'.png'); s=2
    g=cv2.cvtColor(f,6); H,W=g.shape
    gs=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA)
    k=cv2.getStructuringElement(2,(31,31))
    bh=cv2.morphologyEx(gs,cv2.MORPH_BLACKHAT,k); th=cv2.morphologyEx(gs,cv2.MORPH_TOPHAT,k)
    for nm,r in (('bh',bh),('th',th)):
        print(n,nm,np.percentile(r,[50,90,99,99.5,99.9]))
    cv2.imwrite('approaches/_bh_'+n+'.png',cv2.resize(np.hstack([bh,th])*2,(1920,540)))
