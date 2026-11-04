import cv2,numpy as np,sys,time
sys.path.insert(0,'dev/approaches/tmp_ws'); from m2 import binmask
def prep(f,S=2):
    g=cv2.cvtColor(f,6); H,W=g.shape
    b=cv2.erode(binmask(g),cv2.getStructuringElement(2,(31,31)))
    g2=cv2.resize(g,(W//S,H//S),interpolation=cv2.INTER_AREA); b2=cv2.resize(b,(W//S,H//S),interpolation=0)
    h,w=g2.shape; s=4
    bg=cv2.resize(cv2.medianBlur(cv2.resize(g2,(w//s,h//s),interpolation=cv2.INTER_AREA),21//S*2+1 if S>1 else 21),(w,h)).astype(np.float32)
    d=g2.astype(np.float32)-bg; d[b2==0]=0
    pos=np.percentile(d[b2>0],99.9); neg=-np.percentile(d[b2>0],0.1); pol=1 if pos>neg else -1
    return f,g2,b2,d*pol,max(pos,neg)
if __name__=='__main__':
 for n in ['gt27_0','frame_focused']:
    f,g,b,sd,peak=prep(cv2.imread(f'dev/{n}.png'))
    m=((sd>0.4*peak)*255).astype(np.uint8)
    dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
    sk=cv2.ximgproc.thinning(m) if hasattr(cv2,'ximgproc') else None
    print(n,'ximgproc',sk is not None)
    loc=(dt>=cv2.dilate(dt,np.ones((3,3))))&(dt>1.5)
    print(n,'ridge dt hist',np.percentile(dt[loc],[25,50,75,90]))
