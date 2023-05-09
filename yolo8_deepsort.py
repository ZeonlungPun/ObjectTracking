import numpy as np
from ultralytics import YOLO
import cv2,cvzone,math
from DeepSORTTracker import Tracker

cap=cv2.VideoCapture("E:\\UCF11_updated_mpg\\UCF11_updated_mpg\\basketball\\v_shooting_01\\v_shooting_01_01.mpg")
#model=YOLO("../weights/best067.pt")


dectect_thred=0.35

while True:
    success,img=cap.read()
    #results=model(img,stream=True)







    cv2.imshow("image",img)
    cv2.waitKey(1)
