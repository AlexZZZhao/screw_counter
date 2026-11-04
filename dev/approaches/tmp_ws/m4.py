import cv2,numpy as np,sys
sys.path.insert(0,'dev/approaches/tmp_ws'); from m2 import binmask
for n in ['gt27_0','gt27_3','frame_focused','white']:
    f=cv2.imread(f'dev/{n}.png'); g=cv2.cvtColor(f,6); H,W=g.shape
    b=binmask(g); s=8
    sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA); bs=cv2.erode(cv2.resize(b,(W//s,H//s),interpolation=cv2.INTER_NEAREST),np.ones((5,5),np.uint8))
    bgc=cv2.medianBlur(sm,5)
    v=bgc[bs>0].astype(np.float32); t,_=cv2.threshold(v.astype(np.uint8).reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    lo=v[v<=t]; hi=v[v>t]; w0=len(lo)/len(v); sb=w0*(1-w0)*(hi.mean()-lo.mean())**2; eta=sb/v.var()
    print(n,'t',t,'eta',round(eta,3),'means',lo.mean().round(),hi.mean().round(),'frac',round(w0,2))
