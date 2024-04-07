import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/home/kingargroo/corn/runs/detect/train131/weights/best.pt')

# Open the video file
video_path = "/home/kingargroo/YOLOVISION/fishdataset/fish.MOV"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 建立新影片檔案
output_video = cv2.VideoWriter('/home/kingargroo/YOLOVISION/output_video.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (frame_width, frame_height))
#create txt to save
txt_file = open("bot_sort.txt", 'a+')
#initialize frame id
frame_id=-1

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        frame_id+=1
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,tracker="botsort.yaml")
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        all_boxes=results[0].boxes
        predict_class=all_boxes.cls
        predict_id=all_boxes.id
        predict_box=all_boxes.xywh
        predict_score=all_boxes.conf

        for score,box,cls,id in zip(predict_score,predict_box,predict_class,predict_id):
            score, box, cls, id=score.numpy(),box.numpy(),int(cls.numpy()),int(id.numpy())
            cx,cy,w,h=box[0],box[1],box[2],box[3]
            x1,y1=int(cx-w/2),int(cy-h/2)
            w,h=int(w),int(h)
            output_str=str(frame_id)+' '+str(id)+' '+str(x1)+' '+str(y1)+' '+str(w)+' '+str(h)+' '+str(score)+' '+str(-1)+' '+str(-1)+' '+str(-1)+'\n'
            txt_file.write(output_str)

        # Display the annotated frame
        #cv2.imshow("YOLOv8 Tracking", annotated_frame)
        output_video.write(annotated_frame)

    else:
        # Break the loop if the end of the video is reached
        break