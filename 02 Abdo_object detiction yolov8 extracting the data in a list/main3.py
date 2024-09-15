from ultralytics import YOLO
import cv2
import numpy as np
import requests
from io import BytesIO

## 1- Load the YOLOv8 model
model = YOLO('yolov8n.pt')

## 2- Load the image
# img = cv2.imread("00.jpg")
# img=cv2.resize(img,(800,600))

#Define the video source
cap = cv2.VideoCapture("video2.mp4")
while True:
    success,img=cap.read()
    img=cv2.resize(img,(1020,800))
    if success:
        # Step 3: Apply the model
        results = model(img,verbose=False)
        #results = model(img,verbose=False)

        # #Step 4: Extract bounding boxes, confidencesو classes and names
        boxes = results[0].boxes.xyxy.tolist()
        xywhs = results[0].boxes.xywh.tolist()
        confs = results[0].boxes.conf.tolist()
        classes = results[0].boxes.cls.tolist()
        names = model.names

        # Step 5: Iterate through the data and draw bounding boxes and labels
        for xywh, box, conf, cls in zip(xywhs,boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
            xc,yc,w,h = map(int,xywh)
            name = names[int(cls)]  # Get class name from the model's names list
               
            # # Draw a circle at the center
            cv2.circle(img,(xc,yc),4,(255,0,0),-1)
            cv2.rectangle(img, (xc-w//2, yc-h//2), (xc+w//2, yc+h//2), (0, 0, 255), 1)  # Red bounding box
     
            #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
            
            ## Add labels
            cv2.putText(img, f'{name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Label with class name and confidence

            # # Print detected objects info
            print(f"Class: {name}, Confidence: {conf:.2f}, Coordinates: ({x1},{y1},{x2},{y2})")

        # Step 6: Display the image with bounding boxes and labels
        cv2.imshow("Abdo-YOLOv8 Detection", img)

    else:
           break

    if cv2.waitKey(0)&0xFF==27:
            break;
cap.release()
cv2.destroyAllWindows()
