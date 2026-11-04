import cv2,numpy as np,sys
sys.path.insert(0,'dev'); from proto3 import run
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread('dev/'+n+'.png'); m,b=run(f)
    s=4; g=cv2.cvtColor(cv2.resize(f,None,fx=1/s,fy=1/s,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    bs=cv2.resize(b,g.shape[::-1],interpolation=cv2.INTER_NEAREST)
    gb=cv2.GaussianBlur(g,(0,0),1.5)
    lap=cv2.GaussianBlur(np.abs(cv2.Laplacian(gb.astype(np.float32),cv2.CV_32F)),(0,0),2)
    v=gb[bs>0]; h=np.bincount(v,minlength=256); mode=int(np.argmax(cv2.GaussianBlur(h.astype(np.float32).reshape(1,-1),(9,1),0)))
    print(n,'mode',mode, np.percentile(lap[bs>0],[25,50,75]))
    fl=((np.abs(gb.astype(int)-mode)<12)&(bs>0)&(lap<np.percentile(lap[bs>0],75))).astype(np.uint8)
    fl=cv2.morphologyEx(fl,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    nn,lab,st,_=cv2.connectedComponentsWithStats(fl); i=1+np.argmax(st[1:,4])
    cs,_=cv2.findContours((lab==i).astype(np.uint8),0,2); hull=cv2.convexHull(max(cs,key=cv2.contourArea))
    v=cv2.resize(f,g.shape[::-1]); v[lab==i]=(v[lab==i]*.5+np.array([0,128,0])).astype(np.uint8)
    cv2.drawContours(v,[hull],-1,(0,0,255),1)
    cv2.imwrite('dev/approaches/_seg/fl_'+n+'.png',cv2.resize(v,None,fx=2,fy=2))
