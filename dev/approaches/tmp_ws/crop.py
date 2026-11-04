import cv2,sys
n,x0,y0,x1,y1,s=sys.argv[1],*map(int,sys.argv[2:6]),float(sys.argv[6])
f=cv2.imread(f'dev/{n}.png')[y0:y1,x0:x1]; f=cv2.resize(f,None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
cv2.imwrite(sys.argv[7],f)
