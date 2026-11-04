import cv2, numpy as np, sys
f = cv2.imread(sys.argv[1])
x0,y0,x1,y1 = 790,310,1135,925   # inner floor of the bin, eyeballed
roi = f[y0:y1, x0:x1]
g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))
_, m = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(m)
areas = st[1:, cv2.CC_STAT_AREA]
print('blobs', n-1, sorted(areas))
vis = roi.copy()
for i in range(1,n):
    x,y,w,h,a = st[i]
    if a < 150: continue
    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.putText(vis,str(a),(x,y-2),0,0.4,(0,255,255),1)
cv2.imwrite('dev/vis.png', np.hstack([vis, cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)]))
