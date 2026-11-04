import cv2,numpy as np,time
exec(open('dev/approaches/tmp/ex5.py').read().split('    M=cv2.moments')[0].replace('def prep(n):','def prep(n):\n    pass') )
