# Detect objects using yolov8 and open cv

import cv2
import pandas as pd
import numpy as np

from ultralytics import YOLO

#Define the pretrained model
model=YOLO("yolov8n.pt")

#Define the video source
cap = cv2.VideoCapture("video2.mp4")
#img = cv2.imread("image.jpg")

while True:
    success,frame=cap.read()
    if success:
        frame=cv2.resize(frame,(1020,500))
        results = model.predict(frame)
 
       # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2.imshow("Abdo", annotated_frame)
    else:
        break
    if cv2.waitKey(1)&0xFF==27:
        break
    
cap.release()
cv2.destroyAllWindows()
