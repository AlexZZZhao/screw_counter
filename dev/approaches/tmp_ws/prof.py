import cv2,numpy as np
for n,ys in [('gt27_0',[250,450,620,850]),('frame_focused',[300,450,650])]:
    f=cv2.imread(f'dev/{n}.png'); lab=cv2.cvtColor(cv2.GaussianBlur(f,(0,0),3),cv2.COLOR_BGR2Lab)
    for y in ys:
        print(n,y,' '.join(f"{x}:{lab[y,x,0]}/{lab[y,x,1]-128}/{lab[y,x,2]-128}" for x in range(680,1300,20)))
