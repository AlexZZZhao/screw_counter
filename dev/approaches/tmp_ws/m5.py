import cv2,numpy as np,sys
sys.path.insert(0,'dev/approaches/tmp_ws'); from m2 import binmask
def floor(g,b):
    H,W=g.shape; s=8
    sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA); bs=cv2.resize(b,(W//s,H//s),interpolation=cv2.INTER_NEAREST)
    bg=cv2.medianBlur(sm,15)
    v=bg[bs>0]; t,_=cv2.threshold(v.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    x,y,w,h=cv2.boundingRect(bs); c=bg[y+h//3:y+2*h//3,x+w//3:x+2*w//3]
    fl=(bg<=t) if np.mean(c<=t)>0.5 else (bg>t)
    fl=(fl&(bs>0)).astype(np.uint8)
    n,lab,st,_=cv2.connectedComponentsWithStats(fl); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    m=np.zeros_like(sm); cv2.drawContours(m,[hull],-1,255,-1)
    return cv2.resize(m,(W,H),interpolation=cv2.INTER_NEAREST)
if __name__=='__main__':
 for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread(f'dev/{n}.png'); g=cv2.cvtColor(f,6)
    fm=floor(g,binmask(g)); v=f.copy(); cv2.drawContours(v,cv2.findContours(fm,0,2)[0],-1,(0,255,0),3)
    cv2.imwrite(f'dev/approaches/tmp_ws/fl_{n}.png',cv2.resize(v[100:1000,650:1300],None,fx=0.5,fy=0.5))
