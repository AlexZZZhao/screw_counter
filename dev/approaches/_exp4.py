import cv2,numpy as np
for n in ['gt27_0','gt27_3','frame_focused','white']:
    f=cv2.imread(n+'.png'); H,W=f.shape[:2]
    g=cv2.cvtColor(f,6)
    sm=cv2.resize(g,(W//8,H//8),interpolation=cv2.INTER_AREA)
    bg=cv2.medianBlur(sm,9)
    _,br=cv2.threshold(cv2.GaussianBlur(sm,(0,0),1),0,255,8)
    nl,lab,st,cen=cv2.connectedComponentsWithStats(br); i=1+np.argmax(st[1:,4])
    cx,cy=map(int,cen[i])
    outs=[]
    for dlt in (1,2,3):
        m=np.zeros((sm.shape[0]+2,sm.shape[1]+2),np.uint8)
        cv2.floodFill(bg.copy(),m,(cx,cy),0,dlt,dlt,4|cv2.FLOODFILL_MASK_ONLY|(255<<8))
        fl=m[1:-1,1:-1]; outs.append(fl)
        print(n,dlt,(fl>0).sum(), st[i,4])
    v=cv2.cvtColor(sm,8); 
    for c,fl in zip([(0,0,255),(0,255,0),(255,0,0)],outs[::-1]):
        v[fl>0]=(0.5*v[fl>0]+0.5*np.array(c)).astype(np.uint8)
    cv2.imwrite('approaches/_ff_'+n+'.png',cv2.resize(v,(960,540),interpolation=cv2.INTER_NEAREST))
