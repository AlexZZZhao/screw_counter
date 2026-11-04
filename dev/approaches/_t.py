import cv2,time,sys
sys.path.insert(0,'.')
import rodfit
for n in ['gt27_0','gt27_1','frame_focused','white']:
    f=cv2.imread('../'+n+'.png'); t=time.time(); r=rodfit.count_screws(f); dt=time.time()-t
    d=r['debug']; print(n,r['count'],'%.0fms'%(dt*1000),{k:d[k] for k in ('pol','sigma','typ','thr')},d['model'])
    cv2.imwrite('_o_'+n+'.png',cv2.resize(rodfit.draw(f,r),(1440,810)))
