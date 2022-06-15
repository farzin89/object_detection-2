import cv2
import numpy as np
import matplotlib as plt

cap = cv2.VideoCapture("video.mp4")
#cap = cv2.VideoCapture(1)

kernel = None
backgroundobject = cv2.createBackgroundSubtractorMOG2( detectShadows = True)

while True :
    ret,frame = cap.read()
    if not ret :
        break

    fgmask = backgroundobject.apply(frame)

    #perform thresholding to get ride of the shadows.
    _,fgmask = cv2.threshold(fgmask,250,255,cv2.THRESH_BINARY)

    # apply some morphological operations to make sure you have a good mask
    fgmask = cv2.erode(fgmask,kernel,iterations = 1)
    fgmask = cv2.dilate(fgmask,kernel,iterations = 2)

    # detect contours in the frame
    contours,_ = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    frameCopy = frame.copy()

    # loop over each contour found in the frame.
    for cnt in contours:

        #make sure the contour area is somewhat higher than some threshold to make sure its a car and not some noise.
        if cv2.contourArea(cnt) > 400:
            x,y,width,height = cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy,(x,y),(x+width,y+height),(0,0,255),2)
            # write car detection near the bounding box drawn
            cv2.putText(frameCopy,"Car Detected",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1,cv2.LINE_AA)


    # extract the foreground from the frame using the segmented mask.
    foregroundpart = cv2.bitwise_and(frame,frame,mask = fgmask)

    # Stack the original frame, extracted foreground,and annotated frame.
    stacked = np.hstack((frame,foregroundpart,frameCopy))



    cv2.imshow("All three",cv2.resize(stacked,None,fx = 0.5,fy=0.5))
    cv2.imshow("Clean Mask",fgmask)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
