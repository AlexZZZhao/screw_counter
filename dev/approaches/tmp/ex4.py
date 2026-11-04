import cv2,numpy as np
exec(open('dev/approaches/tmp/ex3.py').read().split('for n in')[0])
def prep(n):
    f=cv2.imread('dev/'+n+'.png'); g=cv2.cvtColor(f,6); H,W=g.shape
    bm,sm,s=binmask(g); bms=cv2.resize(bm,sm.shape[::-1],interpolation=0); bg=cv2.medianBlur(sm,9)
    for it in range(2):
        d=cv2.absdiff(sm,bg); fg=((d>12)&(bms>0)).astype(np.uint8)*255
        fg=cv2.dilate(fg,np.ones((3,3)))|(bms==0).astype(np.uint8)*255
        bg=cv2.inpaint(sm,fg,5,cv2.INPAINT_TELEA)
    BG=cv2.resize(bg,(W,H)); d=cv2.absdiff(cv2.GaussianBlur(g,(0,0),1.2),BG); d[cv2.erode(bm,np.ones((15,15)))==0]=0
    return f,d
for n in ['gt27_0','frame_focused']:
    f,d=prep(n)
    vs=[]
    for t in (25,45,70):
        m=(d>t).astype(np.uint8)*255
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        vs.append(cv2.cvtColor(m,8)[250:900,700:1250])
    cv2.imwrite('dev/approaches/tmp/t_'+n+'.png',np.hstack(vs))
