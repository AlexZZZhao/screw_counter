import cv2,numpy as np
f=cv2.imread('gt27_0.png'); g=cv2.cvtColor(f,6).astype(np.float32)
for nm,(y0,y1,x0,x1) in (('r',(280,840,1000,1280)),('l',(280,840,740,920))):
    c=g[y0:y1,x0:x1]
    gx=cv2.Sobel(c,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(c,cv2.CV_32F,0,1,ksize=3)
    gm=cv2.magnitude(gx,gy)
    hp=np.abs(c-cv2.GaussianBlur(c,(0,0),3))
    tiles=[c,np.clip(gm/4,0,255),np.clip(hp*4,0,255)]
    cv2.imwrite('approaches/_hp_'+nm+'.png',np.hstack(tiles).astype(np.uint8))
