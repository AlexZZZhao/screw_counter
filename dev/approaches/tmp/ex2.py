import cv2,numpy as np,sys
sys.path.insert(0,'dev'); from proto3 import run
for n in ['gt27_0','frame_focused']:
    f=cv2.imread('dev/'+n+'.png'); m,b=run(f)
    r=8 if n=='gt27_0' else 7
    o=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
    v=f.copy(); v[m>0]=(v[m>0]*0.5+np.array([0,0,128])).astype(np.uint8); v[o>0]=(0,255,0)
    cv2.imwrite('dev/approaches/tmp/o_'+n+'.png',v[250:900,700:1250])
