import numpy as np
from ultralytics import YOLO
import cv2,cvzone,math
from sort import *

cap=cv2.VideoCapture("E:\\opencv\\yolo8Sort\\Videos\\cars.mp4")
model=YOLO("../weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def test_line(resultTracker,limits,totalCount,img):
    for result in resultTracker:
        x1, y1, x2, y2 ,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.putTextRect(img, f'ID:{int(id)}', (max(0, x1)+50, max(35, y1)),
                           scale=1, offset=3 , thickness=1)
        cx,cy=x1+w//2,y1+h//2

        if limits[0]<cx <limits[2] and limits[1] -20<cy <limits[1]+20:
            totalCount.add(int(id))
    return totalCount




tracker1=Sort(max_age=20,min_hits=2,iou_threshold=0.3)
tracker2=Sort(max_age=20,min_hits=2,iou_threshold=0.3)
limits=[380,297,673,297]
totalCountCars=set()
totalCountTruck=set()

while True:
    success,img=cap.read()
    results=model(img,stream=True)

    detections1=np.empty((0,5))
    detections2 = np.empty((0, 5))


    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),l=9)

            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            if currentClass=="car" or currentClass=="truck" and conf>0.3:

                cvzone.putTextRect(img,f'{classNames[cls]}',(max(0,x1),max(35,y1)),
                               scale=1,offset=3,thickness=1)
                currentArray=np.array([x1,y1,x2,y2,conf])
                if currentClass=="car":
                    detections1=np.vstack((detections1,currentArray))

                elif currentClass=="truck":
                    detections2 = np.vstack((detections2, currentArray))


    resultTracker1=tracker1.update(detections1)
    resultTracker2=tracker2.update(detections2)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    totalCountCars=test_line(resultTracker1,limits,totalCountCars,img)
    totalCountTruck=test_line(resultTracker2,limits,totalCountTruck,img)
    cv2.putText(img, f'car:' + str(len(totalCountCars)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.putText(img, f'trucks' + str(len(totalCountTruck)), (255, 150), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)





    cv2.imshow("image",img)
    key=cv2.waitKey(1)
    if key==27:
        break
