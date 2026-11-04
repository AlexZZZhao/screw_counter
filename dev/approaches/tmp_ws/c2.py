import cv2,numpy as np
f=cv2.imread('dev/gt27_0.png'); g=cv2.cvtColor(f,6).astype(np.float32)
H,W=g.shape; s=4
bg=cv2.resize(cv2.medianBlur(cv2.resize(g.astype(np.uint8),(W//s,H//s)),15),(W,H)).astype(np.float32)
d=g-bg
for name,(x0,y0,x1,y1) in {'ghostL':(760,300,800,450),'realL':(815,300,860,430),'ghostR':(1180,350,1220,560),'realR':(1130,480,1175,580)}.items():
    r=d[y0:y1,x0:x1]; print(name, np.percentile(r,[1,5,10,50]).round(1))
# saturation
hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
for name,(x0,y0,x1,y1) in {'ghostL':(760,300,800,450),'realL':(815,300,860,430),'floor':(950,250,1050,300)}.items():
    print(name,'S',np.percentile(hsv[y0:y1,x0:x1,1],[50,90,99]),'B-R', np.percentile(f[y0:y1,x0:x1,0].astype(int)-f[y0:y1,x0:x1,2],[1,50,99]))
