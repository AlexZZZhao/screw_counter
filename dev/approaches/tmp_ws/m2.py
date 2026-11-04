import cv2,numpy as np,sys,time
def binmask(g):
    gb=cv2.GaussianBlur(g,(0,0),3)
    _,br=cv2.threshold(gb,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    br=cv2.morphologyEx(br,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51)))
    n,lab,st,_=cv2.connectedComponentsWithStats(br)
    i=1+np.argmax(st[1:,4]); cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2)
    hull=cv2.convexHull(max(cs,key=cv2.contourArea)); m=np.zeros_like(g); cv2.drawContours(m,[hull],-1,255,-1); return m
if __name__=="__main__":
 for n in ['gt27_0','frame_focused','white']:
     f=cv2.imread(f'dev/{n}.png'); g=cv2.cvtColor(f,6); H,W=g.shape
     b=binmask(g)
     s=8; sm=cv2.resize(g,(W//s,H//s),interpolation=cv2.INTER_AREA); bs=cv2.resize(b,(W//s,H//s),interpolation=cv2.INTER_NEAREST)
     bgc=cv2.medianBlur(sm,11)
     v=bgc[bs>0]; t,_=cv2.threshold(v.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
     lo=v[v<=t]; hi=v[v>t]; print(n,'otsu',t,'means',lo.mean().round(),hi.mean().round(),'frac',len(lo)/len(v))
     out=np.zeros((H//s,W//s*2),np.uint8); out[:,:W//s]=bgc; out[:,W//s:]=((bgc<=t)&(bs>0))*255
     cv2.imwrite(f'dev/approaches/tmp_ws/bg_{n}.png',out)
 