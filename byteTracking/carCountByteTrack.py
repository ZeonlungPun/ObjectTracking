from ultralytics import YOLO
import cv2,cvzone,os
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.tools.detections import Detections
from typing import List
import numpy as np
from supervision.video.source import get_video_frames_generator
from supervision.video.dataclasses import VideoInfo



@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = True


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

cap=cv2.VideoCapture("E:\\opencv\\yolo8Sort\\Videos\\cars.mp4")
model=YOLO("../weights/yolov8l.pt")
model.fuse()

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





# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
        detections: Detections,
        tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

SOURCE_VIDEO_PATH="E:\\opencv\\yolo8Sort\\Videos\\cars.mp4"
# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)


# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
dectect_thred=0.35
CLASS_ID=[1,2,3,5,7]
limits=[400,297,673,297]
totalCount=set()
from tqdm import tqdm

while True:
    for frame in tqdm(generator, total=video_info.total_frames):
        results = model(frame)

        imgGraphics=cv2.imread('E:\\opencv\\yolo8Sort\\graphics.png',cv2.IMREAD_UNCHANGED)
        frame=cvzone.overlayPNG(frame,imgGraphics,(0,0))
        cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)



        for xyxy, conf, class_id, tracker_id in detections:
            x1, y1, x2, y2 = xyxy

            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1

            currentClass=classNames[class_id]

            if currentClass=="car" or currentClass=="truck" or currentClass=="bus"\
                or currentClass=="motorbike" and conf>dectect_thred:
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9)
                cvzone.putTextRect(frame,f'{classNames[class_id]}',(max(0,x1),max(35,y1)),
                               scale=1,offset=3,thickness=1)


            cvzone.putTextRect(frame, f'ID:{int(tracker_id)}', (max(0, x1)+50, max(35, y1)),
                                   scale=1, offset=3 , thickness=1)
            cx,cy=x1+w//2,y1+h//2


            if limits[0]<cx <limits[2] and limits[1] -20<cy <limits[1]+20:
                    totalCount.add(int(tracker_id))
            cv2.putText(frame,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

        cv2.imshow("image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break




