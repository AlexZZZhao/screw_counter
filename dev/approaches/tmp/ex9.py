import cv2,numpy as np,sys,time
sys.path.insert(0,'dev/approaches/tmp'); from common import contrast
def canon(d,m,lab,i,st):
    x,y,w,h,a=st[i]; pad=10
    x0,y0=max(0,x-pad),max(0,y-pad); x1,y1=x+w+pad,y+h+pad
    cm=(lab[y0:y1,x0:x1]==i).astype(np.uint8)
    cs,_=cv2.findContours(cm,0,2); (cx,cy),(rw,rh),ang=cv2.minAreaRect(max(cs,key=cv2.contourArea))
    if rw>rh: ang+=90; rw,rh=rh,rw
    L=int(rh)+20; Wd=int(rw)+20
    M=cv2.getRotationMatrix2D((cx+x0,cy+y0),ang,1.0); M[0,2]+=Wd/2-(cx+x0); M[1,2]+=L/2-(cy+y0)
    mm=cv2.warpAffine((lab==i).astype(np.uint8),M,(Wd,L),flags=0)
    dd=cv2.warpAffine(d,M,(Wd,L),flags=1)
    return mm,dd,rw,rh,ang,(cx+x0,cy+y0)
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread('dev/'+n+'.png'); g,d,bm=contrast(f)
    t,_=cv2.threshold(d[bm>0],0,255,8)
    m=(d>t).astype(np.uint8); m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    nn,lab,st,_=cv2.connectedComponentsWithStats(m)
    for i in range(1,nn):
        if st[i,4]<300: continue
        mm,dd,rw,rh,ang,c=canon(d,m,lab,i,st)
        if rh/rw<2.2: continue
        prof=mm.sum(1); prof=prof[prof>0]
        k=max(2,len(prof)//6)
        print(n,i,int(c[0]),int(c[1]),'L',len(prof),'w',int(rw),'ends',round(prof[:k].mean(),1),round(prof[-k:].mean(),1),'mid',np.median(prof))
