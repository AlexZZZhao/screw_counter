import cv2,numpy as np
src=open('approaches/_exp6.py').read()
body=src.split("    v=c[floor")[0]
body=body.replace("for n in ['gt27_0','frame_focused','white']:","def F(n):")
body+="    return f,g,c,fl,hull,pol\n"
exec(body)
for n in ['gt27_0','frame_focused']:
    f,g,c,fl,hull,pol=F(n)
    pts=np.argwhere(fl)[:,::-1].astype(np.float32)*4
    r=cv2.minAreaRect(pts); box=cv2.boxPoints(r)
    vis=cv2.cvtColor(g.astype(np.uint8),8)
    for t,col in ((25,(0,200,255)),(50,(0,0,255))):
        vis[c>t]=col
    cv2.polylines(vis,[(box/2).astype(np.int32)],True,(0,255,0),1)
    cv2.imwrite('approaches/_t_'+n+'.png',vis)
    pts2=(box//2).astype(np.int32); fm=np.zeros(g.shape,np.uint8); cv2.fillPoly(fm,[pts2],1)
    v=c[(fm>0)]; lo,_=cv2.threshold(np.clip(v,0,255).astype(np.uint8).reshape(-1,1),0,255,8)
    w=v[v>lo]; hi,_=cv2.threshold(np.clip(w,0,255).astype(np.uint8).reshape(-1,1),0,255,8)
    print(n,'lo',lo,'hi',hi,np.histogram(w,bins=12,range=(lo,200))[0])
    if n=='gt27_0':
        cc=c[140:420,500:640]; 
        tiles=[np.clip(cc*1.5,0,255).astype(np.uint8)]+[((cc>t)*255).astype(np.uint8) for t in (35,50,65,80)]
        cv2.imwrite('approaches/_rt.png',cv2.resize(np.hstack(tiles),None,fx=1.5,fy=1.5,interpolation=cv2.INTER_NEAREST))
        cc=c[140:420,370:460]
        tiles=[np.clip(cc*1.5,0,255).astype(np.uint8)]+[((cc>t)*255).astype(np.uint8) for t in (35,50,65,80)]
        cv2.imwrite('approaches/_lt.png',cv2.resize(np.hstack(tiles),None,fx=1.5,fy=1.5,interpolation=cv2.INTER_NEAREST))
    vv=c[(fm>0)]; mad=np.median(np.abs(vv-np.median(vv)))*1.4826
    fgv=vv[vv>6*mad]
    print(n,'mad',mad,'fg pct',np.percentile(fgv,[25,50,75,90,95]), len(fgv))
