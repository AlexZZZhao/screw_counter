import cv2,numpy as np,sys
sys.path.insert(0,'dev/approaches/tmp'); from common import contrast
for n in ['gt27_0','frame_focused','white']:
    f=cv2.imread('dev/'+n+'.png'); g,d,bm=contrast(f)
    t,_=cv2.threshold(d[bm>0],0,255,8)
    m=(d>t).astype(np.uint8)*255; m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    nn,lab,st,_=cv2.connectedComponentsWithStats(m)
    rows=[]
    for i in range(1,nn):
        if st[i,4]<100: continue
        cs,_=cv2.findContours((lab[st[i,1]:st[i,1]+st[i,3],st[i,0]:st[i,0]+st[i,2]]==i).astype(np.uint8),0,2)
        (cx,cy),(w,h),a=cv2.minAreaRect(max(cs,key=cv2.contourArea))
        w,h=min(w,h),max(w,h); rows.append((st[i,4],round(w),round(h),round(st[i,4]/(w*h),2)))
    rows.sort(); print(n,'otsu',t); print(rows)
