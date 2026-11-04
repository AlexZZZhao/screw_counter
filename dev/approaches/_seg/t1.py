import cv2,numpy as np,sys
sys.path.insert(0,'dev'); from proto3 import run
for n in ['gt27_0','frame_focused']:
    f=cv2.imread('dev/'+n+'.png'); m,b=run(f)
    v=f.copy(); v[m>0]=(0,0,255); cv2.drawContours(v,cv2.findContours(b,0,2)[0],-1,(0,255,0),3)
    cv2.imwrite('dev/approaches/_seg/m_'+n+'.png',cv2.resize(v,(960,540)))
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    lsd=cv2.createLineSegmentDetector()
    l=lsd.detect(g)[0]; print(n,len(l))
