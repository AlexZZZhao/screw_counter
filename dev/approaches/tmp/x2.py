import cv2,numpy as np
f=cv2.imread('gt27_0.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
c=f[280:800,740:880]; cv2.imwrite('approaches/tmp/c_left.png',cv2.resize(c,None,fx=2,fy=2))
print('ghost row y=400 x740-880', g[400,740:880:3])
print('real screw row y=400 x 810-870', g[400,810:870])
c=f[330:800,1100:1240]; cv2.imwrite('approaches/tmp/c_right.png',cv2.resize(c,None,fx=2,fy=2))
print('row 550 x1120-1230', g[550,1120:1230:2])
