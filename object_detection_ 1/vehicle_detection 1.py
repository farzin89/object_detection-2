import cv2
import numpy as np
import matplotlib as plt

cap = cv2.VideoCapture("video.mp4")

backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

while(1) :
    ret,frame = cap.read()
    if not ret :
        break

    fgmask = backgroundobject.apply(frame)
    real_part = cv2.bitwise_and(frame,frame,mask = fgmask)
    fgmask_3 = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)

    stacked = np.hstack((fgmask_3,frame,real_part))
    cv2.imshow("All three",cv2.resize(stacked,None,fx = 0.65,fy=0.65))

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
