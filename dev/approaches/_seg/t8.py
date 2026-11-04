import cv2,numpy as np,sys
sys.path.insert(0,'dev'); from proto3 import run
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread('dev/'+n+'.png'); _,b=run(f)
    h=cv2.resize(f,(960,540),interpolation=cv2.INTER_AREA); g=cv2.cvtColor(h,cv2.COLOR_BGR2GRAY)
    bs=cv2.resize(b,(960,540),interpolation=cv2.INTER_NEAREST)
    bg=cv2.resize(cv2.medianBlur(cv2.resize(g,(480,270),interpolation=cv2.INTER_AREA),15),(960,540))
    d=cv2.absdiff(g,bg); v=d[(bs>0)&(d>8)]
    t,_=cv2.threshold(v.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(n,'otsu',t,'p99',np.percentile(d[bs>0],99.5))
    for T in (t,):
        m=((d>T)&(bs>0)).astype(np.uint8)*255
        vv=h.copy(); vv[m>0]=(0,0,255); cv2.imwrite('dev/approaches/_seg/o_'+n+'.png',vv)
