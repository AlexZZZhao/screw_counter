import sys,cv2,numpy as np; sys.path.insert(0,'.')
import skeleton as S
for n in ['gt27_0','frame_focused','white']:
    m,fl,d=S._segment(cv2.imread('../%s.png'%n))
    dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
    lm=(dt>=cv2.dilate(dt,np.ones((3,3))))&(dt>0)
    print(n,np.histogram(dt[lm],bins=[0,1.1,1.5,2.1,2.6,3.1,3.6,4.1,5,6,8,12])[0])
    for k in (3,5):
        o=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
        cv2.imwrite('tmp/op%d_%s.png'%(k,n),o[25:500,320:680]*255)
