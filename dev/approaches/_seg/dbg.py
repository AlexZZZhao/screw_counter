import sys,cv2,numpy as np; sys.path.insert(0,'dev/approaches'); import segments as S
n=sys.argv[1]; x0,y0,x1,y1=map(int,sys.argv[2:6])
f=cv2.imread('dev/'+n+'.png'); r=S.count_screws(f); d=r['debug']
g=cv2.resize(f,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
v=(g*0.5).astype(np.uint8); v[d['fg']>0]=(v[d['fg']>0]*0.5+np.array([60,60,60])).astype(np.uint8)
segs=S._lsd().detect(cv2.cvtColor(g,cv2.COLOR_BGR2GRAY))[0].reshape(-1,4)
z=4; v=cv2.resize(v[y0:y1,x0:x1],None,fx=z,fy=z,interpolation=cv2.INTER_NEAREST)
P=lambda p:(int((p[0]-x0)*z),int((p[1]-y0)*z))
for a,b,c,e in segs: cv2.line(v,P((a,b)),P((c,e)),(255,0,0),1)
for it,k in zip(r['items'],r['debug']['per_rod']):
    pass
for it in r['items']:
    p0=np.array(it['p0'])/2;p1=np.array(it['p1'])/2
    cv2.line(v,P(p0),P(p1),(0,255,0) if it['n']==1 else (0,140,255),2)
for (a,b),L,k in zip(d.get('rods',[]),d['lens'],d['per_rod']):
    if k==0: cv2.line(v,P(a),P(b),(0,0,255),2)
cv2.imwrite('dev/approaches/_seg/dbg.png',v)
