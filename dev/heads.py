import cv2, numpy as np, sys
def heads(roi, min_r=6, off=0):
    g=cv2.GaussianBlur(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),(5,5),0)
    th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(61,61)))
    o,_=cv2.threshold(th,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,m=cv2.threshold(th,max(o,25)+off,255,cv2.THRESH_BINARY)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    H,W=m.shape
    n,lab,st,_=cv2.connectedComponentsWithStats(m)
    for i in range(1,n):
        x,y,w,h,a=st[i]
        edge = x<=2 or y<=2 or x+w>=W-2 or y+h>=H-2
        fill = a/(w*h)
        if edge and (max(w,h)/max(1,min(w,h))>4 or max(w,h)>0.5*min(H,W)):
            m[lab==i]=0
    d=cv2.distanceTransform(m,cv2.DIST_L2,5)
    ys,xs=np.where((d==cv2.dilate(d,np.ones((3,3))))&(d>=min_r))
    cand=sorted(zip(d[ys,xs],xs,ys),reverse=True)
    keep=[]
    for r,x,y in cand:
        if all((x-kx)**2+(y-ky)**2 > (1.6*max(r,kr))**2 for kr,kx,ky in keep):
            keep.append((r,x,y))
    return keep, m
if __name__=='__main__':
    f=cv2.imread(sys.argv[1]); x0,y0,w,h=map(int,sys.argv[2].split(','))
    roi=f[y0:y0+h,x0:x0+w]
    for r in (5,6,7):
        k,m=heads(roi,r); print('min_r',r,'count',len(k))
    k,m=heads(roi,6)
    v=roi.copy()
    for r,x,y in k: cv2.circle(v,(int(x),int(y)),int(r),(0,0,255),2)
    cv2.imwrite('dev/heads.png',np.hstack([cv2.resize(v,None,fx=1.4,fy=1.4),cv2.resize(cv2.cvtColor(m,cv2.COLOR_GRAY2BGR),None,fx=1.4,fy=1.4)]))
