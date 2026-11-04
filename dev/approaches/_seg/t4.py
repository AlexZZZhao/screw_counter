import cv2,numpy as np
f=cv2.imread('dev/gt27_0.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
H,W=g.shape; s=4; bg=cv2.resize(cv2.medianBlur(cv2.resize(g,(W//s,H//s)),15),(W,H))
d=cv2.absdiff(g,bg)
for name,(y0,y1,x0,x1) in {'ghostL':(300,450,755,800),'ghostR':(350,700,1170,1220),'screw':(300,450,810,860),'screwR':(450,700,1130,1170)}.items():
    r=d[y0:y1,x0:x1]; print(name, np.percentile(r,[50,90,99,100]), 'B-R', np.mean(f[y0:y1,x0:x1,0].astype(int)-f[y0:y1,x0:x1,2]))
cv2.imwrite('dev/approaches/_seg/c4.png',cv2.resize(f[250:800,720:860],None,fx=2,fy=2))
