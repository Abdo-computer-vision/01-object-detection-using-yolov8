# Detect objects using yolov8 and open cv

import cv2
import pandas as pd
import numpy as np

from ultralytics import YOLO

#Define the pretrained model
model=YOLO("yolov8n.pt")

#define classes
myFile = open("coco_classes.txt","r")
classNames = myFile.read().split("\n")
#print(classList)

#Define the video source
cap = cv2.VideoCapture("video2.mp4")
#img = cv2.imread("image.jpg")
count=0

while True:
    success,frame=cap.read()
    
    if success:
        frame=cv2.resize(frame,(1020,500))
        results = model.predict(frame,verbose=False)
        
        #------------------------------------------
#         # To print boxes, confidences, and class predictions from the 'results' object:
#         if results[0].boxes is not None:
#             boxes = results[0].boxes  # Extract boxes from the first image result (if batch is used)
#             for box in boxes:
#                 print("Bounding Box (xyxy):", box.xyxy.numpy())  # Coordinates (x1, y1, x2, y2)
#                 print("Confidence:", box.conf.numpy())           # Confidence score for this prediction
#                 print("Class:", box.cls.numpy())                 # Predicted class index
#            
          #---------------------------------------
        # Extract bounding boxes, classes, and confidences
        #print('@@@@@@  results[0].boxes @@@@@@@@@')
        #print(results[0].boxes)
        # results[0].boxes.data contains all the necessary information
        boxes_data = results[0].boxes.data.numpy()  # Convert tensor to numpy array

        # Create a DataFrame from the data
        # The columns could be x1, y1, x2, y2, confidence, class
        df = pd.DataFrame(boxes_data, columns=["x1", "y1", "x2", "y2", "confidence", "class"])

         # Print the DataFrame to verify
        #print(df)
        #print("---------------------------------")

        # You can now perform operations like filtering based on confidence, etc.
        filtered_df = df[df['confidence'] >0.6]
        print(filtered_df)
        print("*********************************")

#--------------------------------------
        # Iterate over each row in the DataFrame to draw rectangles
        for index, row in filtered_df.iterrows():
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            confidence = row['confidence']
            class_id = int(row['class'])
            
            # Draw a rectangle on the image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green box with thickness 2

            # Put class ID and confidence on the box
            label = classNames[class_id]
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Blue text

        cv2.imshow("Abdo", frame)
    else:
        break
    if cv2.waitKey(1)&0xFF==27:
        break
    
    
cap.release()
cv2.destroyAllWindows()
