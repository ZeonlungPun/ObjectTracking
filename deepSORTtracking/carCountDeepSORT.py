from ultralytics import YOLO
import cv2,cvzone,math,os
from DeepSORTTracker import Tracker
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

tracker=Tracker()
dectect_thred=0.35

limits=[400,297,673,297]
totalCount=set()

while True:
    success,img=cap.read()
    results=model(img,stream=True)

    detections=[]
    imgGraphics=cv2.imread('E:\\opencv\\yolo8Sort\\graphics.png',cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(img,imgGraphics,(0,0))
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1


            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            if currentClass=="car" or currentClass=="truck" or currentClass=="bus"\
                or currentClass=="motorbike" and conf>dectect_thred:
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                cvzone.putTextRect(img,f'{classNames[cls]}',(max(0,x1),max(35,y1)),
                               scale=1,offset=3,thickness=1)
                detections.append([x1,y1,x2,y2,conf])
        tracker.update(img,detections)
        for result in tracker.tracks:

            x1, y1, x2, y2=result.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            id=result.track_id
            cvzone.putTextRect(img, f'ID:{int(id)}', (max(0, x1)+50, max(35, y1)),
                               scale=1, offset=3 , thickness=1)
            cx,cy=x1+w//2,y1+h//2
            if limits[0]<cx <limits[2] and limits[1] -20<cy <limits[1]+20:
                totalCount.add(int(id))
        cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)


    cv2.imshow("image",img)
    key=cv2.waitKey(1)
    if key==27:
        break
