import cv2
import numpy as np
from cartoonize import caart
from carton import carton
from neon import neon

videoCaptureObject = cv2.VideoCapture(0)

out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, (720, 1280))
result = True
colorNum=0
while(result):
    ret,img = videoCaptureObject.read()
    # img=caart(img)
    img=carton(img)
    img=neon(img, colorNum)
    colorNum+=1
    if(colorNum == 31):
        colorNum=0
    cv2.imshow("original",np.array(img))
    out.write(img)
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break
videoCaptureObject.release()
cv2.destroyAllWindows()