import cv2,numpy as np
def dmap(g):
    H,W=g.shape
    s8=cv2.resize(g,(W//8,H//8),interpolation=cv2.INTER_AREA)
    bg=cv2.resize(cv2.medianBlur(s8,13),(W,H))
    return cv2.absdiff(cv2.GaussianBlur(g,(0,0),1.5),bg)
g=cv2.cvtColor(cv2.imread('gt27_0.png'),cv2.COLOR_BGR2GRAY)
d=dmap(g)
print('ghost L', d[300:440,760:800].max(), np.percentile(d[300:440,760:800],[50,90,99]))
print('ghost L2', d[500:620,765:815].max(), np.percentile(d[500:620,765:815],[50,90,99]))
print('ghost R', d[350:470,1175:1225].max(), np.percentile(d[350:470,1175:1225],[50,90,99]))
print('ghost R2', d[620:760,1185:1220].max())
print('real', d[300:440,815:860].max(), np.percentile(d[300:440,815:860],[50,90,99]))
g=cv2.cvtColor(cv2.imread('frame_focused.png'),cv2.COLOR_BGR2GRAY); d=dmap(g)
cv2.imwrite('approaches/tmp/dfoc_full.png',cv2.convertScaleAbs(d,alpha=2)[550:850,750:1200])
for t in (30,40,50,60): print(t,(d>t).sum())
