from ultralytics import YOLO
import cv2,cvzone,os
from tracker.mc_bot_sort import STrack,BoTSORT
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.tools.detections import Detections
from typing import List
import numpy as np
from supervision.video.source import get_video_frames_generator
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink


@dataclass(frozen=True)
class BotSORTArgs:
    name:str='exp'
    # tracking args
    track_high_thresh: float=0.3
    track_low_thresh: float=0.05
    track_buffer: int = 30
    new_track_thresh:float=0.4
    track_thresh: float = 0.25
    match_thresh: float = 0.7
    aspect_ratio_thresh: float = 1.3
    min_box_area: float = 10
    fuse_score:bool=False
    mot20:bool=False
    #CMC
    cmc_method: str ="sparseOptFlow"
    # ReID
    with_reid:bool =False
    fast_reid_config:str =r"fast_reid/configs/MOT17/sbs_S50.yml"
    fast_reid_weights:str=r"pretrained/mot17_sbs_S50.pth"
    proximity_thresh:float=0.5
    appearance_thresh:float=0.35
    jde:bool = False
    ablation:bool=False
    device:str='cuda:0'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model=YOLO("/home/kingargroo/corn/runs/detect/train131/weights/best.pt")
model.fuse()

classNames = ["fish"]




# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis],
        detections.class_id[:,np.newaxis]

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

SOURCE_VIDEO_PATH="/home/kingargroo/YOLOVISION/fishdataset/fish.MOV"
# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)


# create BOTSORT instance
BOTSORT_tracker=BoTSORT(BotSORTArgs(),frame_rate=30)
dectect_thred=0.35
CLASS_ID=[0]
limits=[400,297,673,297]
fsishIDset=set()
TARGET_VIDEO_PATH = f"outputfish.mp4"
from tqdm import tqdm

with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(generator, total=video_info.total_frames):
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # tracking detections
        tracks = BOTSORT_tracker.update(
            output_results=detections2boxes(detections=detections),img=frame )
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

            if  conf>dectect_thred:
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9)
                cvzone.putTextRect(frame,f'{classNames[class_id]}',(max(0,x1),max(35,y1)),
                               scale=1,offset=3,thickness=1)


                cvzone.putTextRect(frame, f'ID:{int(tracker_id)}', (max(0, x1)+50, max(35, y1)),
                                   scale=1, offset=3 , thickness=1)
                fsishIDset.add(int(tracker_id))
                #print(fsishIDset)
            cx,cy=x1+w//2,y1+h//2

            #if limits[0]<cx <limits[2] and limits[1] -20<cy <limits[1]+20:
            cv2.putText(frame,str(len(fsishIDset)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
        sink.write_frame(frame)