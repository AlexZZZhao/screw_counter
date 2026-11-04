import cv2, numpy as np, sys
f=cv2.imread(sys.argv[1]); H,W=f.shape[:2]
g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
# 1. bin = largest bright region, holes filled, convex hull, eroded off the rim
gb=cv2.GaussianBlur(g,(0,0),3)
_,bright=cv2.threshold(gb,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
bright=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51)))
n,lab,st,_=cv2.connectedComponentsWithStats(bright)
i=1+np.argmax(st[1:,4]); cs,_=cv2.findContours((lab==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
hull=cv2.convexHull(max(cs,key=cv2.contourArea))
binm=np.zeros_like(g); cv2.drawContours(binm,[hull],-1,255,-1)
binm=cv2.erode(binm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
# 2. local background via large median; objects = darker than bg (blackhat-ish)
bg=cv2.medianBlur(cv2.resize(g,(W//4,H//4)),15); bg=cv2.resize(bg,(W,H))
diff=cv2.subtract(bg,g)  # positive where darker than surroundings
diff[binm==0]=0
for t in (30,45,60):
    m=((diff>t)*255).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    print('t',t,'dark area',int((m>0).sum()))
t=45; m=((diff>t)*255).astype(np.uint8); m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
v=f.copy(); v[m>0]=(0,0,255); cv2.drawContours(v,[hull],-1,(0,255,0),3)
cv2.imwrite('dev/p2.png',cv2.resize(v[300:900,680:1280],None,fx=1,fy=1))
cv2.imwrite('dev/p2diff.png',cv2.resize(diff,None,fx=0.5,fy=0.5))
m=((diff>30)*255).astype(np.uint8)
m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
n,lab,st,cen=cv2.connectedComponentsWithStats(m)
rows=[]
v=f.copy()
for i in range(1,n):
    x,y,w,h,a=st[i]
    if a<150: continue
    cs,_=cv2.findContours((lab==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    (cx,cy),(rw,rh),ang=cv2.minAreaRect(cs[0])
    L=max(rw,rh); Wd=min(rw,rh)
    rows.append((a,round(L),round(Wd)))
    cv2.rectangle(v,(x,y),(x+w,y+h),(0,0,255),1); cv2.putText(v,f"{a}/{L:.0f}",(x,y-2),0,0.35,(255,0,0),1)
print(sorted(rows))
cv2.imwrite('dev/p3.png',cv2.resize(v[300:900,680:1280],None,fx=1.3,fy=1.3))
