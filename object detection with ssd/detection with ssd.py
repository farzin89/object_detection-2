
import cv2

img = cv2.imread('farzin.jpg.jpg')

classNames =[]
classfile = 'coco.names'

with open (classfile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

confipath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightspath,confipath)
net.setInputSize(320,328)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

classIds,confs,bbox = net.detect(img,confThreshold = 0.5)

print(classIds,bbox)

for classId,confidence , box in zip (classIds.flatten(),confs.flatten(),bbox):
    cv2.rectangle(img,box,color =(0,255,0),thickness=3)
    cv2.putText(img,classNames[classId-1],(box[0]+30,box[1]+50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)




cv2.imshow("output",img)
cv2.waitKey(0)