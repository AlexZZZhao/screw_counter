import sys,cv2,numpy as np; sys.path.insert(0,'dev/approaches'); import segments as S
import re
src=open('dev/approaches/segments.py').read()
F={n:cv2.imread('dev/'+n+'.png') for n in ['gt27_0','gt27_1','gt27_2','gt27_3','gt27_4','frame_focused','white']}
def run(cov,gap,claim):
    s=src.replace("cov >= 0.99","cov >= %s"%cov).replace("if gap > W:","if gap > %s * W:"%gap).replace("round(1.5 * W)))","round(%s * W)))"%claim)
    ns={}; exec(compile(s,'x','exec'),ns)
    out=[ns['count_screws'](F[n])['count'] for n in F]
    u=ns['calibrate'](F['gt27_0'],27); cal=[ns['count_screws'](F[n],u)['count'] for n in ['gt27_1','gt27_2','gt27_3','gt27_4']]
    return out,cal
for cov in ('0.99','0.66'):
  for gap in ('1','2.5'):
    for claim in ('1.5','2'):
      print(cov,gap,claim,*run(cov,gap,claim))
