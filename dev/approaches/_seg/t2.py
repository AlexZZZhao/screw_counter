import cv2,numpy as np
f=cv2.imread('dev/gt27_0.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
# row profile across y=540 (full res)
for y in (400,540,700):
  print(y, g[y,700:1300:15].tolist())
h=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
print('sat', h[540,700:1300:30,1].tolist())
f=cv2.imread('dev/frame_focused.png'); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
for y in (300,500,700):
  print('F',y, g[y,650:1300:15].tolist())
h=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
print('sat', h[500,650:1300:30,1].tolist())
