import cv2
import torch
import numpy as np



model = torch.hub.load('ultralytics/yolov5', 'custom',path, force_reload=True)

cap=cv2.VideoCapture()
count=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,600))
    
    results=model(frame)
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
