import cv2, numpy as np, sys
def run(f, t=35, erode=31):
    H,W=f.shape[:2]; g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    gb=cv2.GaussianBlur(g,(0,0),3)
    _,bright=cv2.threshold(gb,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bright=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51)))
    n,lab,st,_=cv2.connectedComponentsWithStats(bright)
    i=1+np.argmax(st[1:,4]); cs,_=cv2.findContours((lab==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    binm=np.zeros_like(g); cv2.drawContours(binm,[hull],-1,255,-1)
    binm=cv2.erode(binm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erode,erode)))
    s=4; bg=cv2.resize(cv2.medianBlur(cv2.resize(g,(W//s,H//s)),15),(W,H))
    d=cv2.absdiff(g,bg); d[binm==0]=0
    m=((d>t)*255).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    return m, binm
if __name__=='__main__':
    for p in sys.argv[1:]:
        f=cv2.imread(p)
        for t in (25,35,45):
            m,b=run(f,t); print(p,t,int((m>0).sum()))
