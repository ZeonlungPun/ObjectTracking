#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include "BYTETracker.h"
#include "inference.h"

using namespace cv;
using namespace std;
using namespace dnn;


static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void generate_yolov8_proposals(std::vector<Detection>& output,std::vector<Object>& proposals)
{
    for (int i=0; i<output.size(); i++)
    {
        Detection detection = output[i];
        Rect box = detection.box;
        float prob=detection.confidence;
        int label=detection.class_id;

        float x0=box.x;
        float y0=box.y;
        float w=box.width;
        float h=box.height;
        

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = label;
        obj.prob = prob;

        proposals.push_back(obj);

    }
}


void DetectWithYolov8(Mat& frame,std::vector<Object>& objects)
{
    Inference inf("/home/punzeonlung/CPP/BytetrackYolov8/yolov8m.onnx", Size(1280,1280), false);
    std::vector<Detection> output = inf.runInference(frame);
    //std::cout << "Number of detections:" << output.size() << std::endl;
    
    generate_yolov8_proposals(output, objects);
    qsort_descent_inplace(objects);

}




int main(int argc, char** argv)
{
    

    const char* videopath = "/home/punzeonlung/CPP/ByteTrack/people.mp4";
    VideoCapture cap(videopath);
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << ", fps: "<<fps<<endl;

    VideoWriter writer("demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(img_w, img_h));

    Mat img;
    BYTETracker tracker(fps, 30);
    int num_frames = 0;
    int total_ms = 1;
	while (true)
    {
        if(!cap.read(img))
            break;
        num_frames++;
        

        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
		if (img.empty())
			break;
        //cout<<"Processing frame "<<num_frames<<endl;
        std::vector<Object> objects;
        auto start = chrono::system_clock::now();
        DetectWithYolov8(img, objects);

        vector<STrack> output_stracks = tracker.update(objects);
        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();
        for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			
            Scalar s = tracker.get_color(output_stracks[i].track_id);
            putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                    0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
            rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			
		}
        putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, (int)output_stracks.size()),
                Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);
        
    }
    cap.release();
    cout << "FPS: " << num_frames * 1000000 / total_ms << endl;

    return 0;
}
