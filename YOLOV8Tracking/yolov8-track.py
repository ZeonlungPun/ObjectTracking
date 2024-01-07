from ultralytics import YOLO
import cv2
import supervision as sv

model=YOLO("E:\opencv\yolo8Sort\weights\yolov8l.pt")
#cap=cv2.VideoCapture("E:\\opencv\\yolo8Sort\\Videos\\cars.mp4")

box_annotator=sv.BoxAnnotator(thickness=2,text_thickness=1,text_scale=0.5)

totalCount=set()
for result in model.track(source="E:\\opencv\\yolo8Sort\\Videos\\cars.mp4",show=False,stream=True):

    frame=result.orig_img
    detections=sv.Detections.from_yolov8(result)
    if result.boxes.id is not None:
        detections.tracker_id=result.boxes.id.cpu().numpy().astype(int)
    labels=[]
    for _, conf, class_id, tracker_id in detections:

        labels.append(f'{tracker_id}{model.model.names[class_id]}{conf:0.2f}')
        totalCount.add(tracker_id)

    frame=box_annotator.annotate(scene=frame,detections=detections,labels=labels)
    cv2.putText(frame, 'totalCounts:'+str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow("image", frame)
    key = cv2.waitKey(10)
    if key==27:
        break




