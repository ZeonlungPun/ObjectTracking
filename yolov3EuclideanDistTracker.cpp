#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <opencv2\opencv.hpp>
#include<iostream>
#include <algorithm>
#include <time.h>
#include<vector>
#include<string>
#include<fstream>
using namespace std;
using namespace cv;


class EuclideanDistTracker
{
public:
	int id_count = 0;
	map<int, Point>center_points;

	map<int, Rect> update(vector<Rect> objects_rect)
	{
		//objects_boxes and ids
		map<int, Rect>objects_bbs_ids;
		//get the center point of a new objects
		for (int i = 0; i < objects_rect.size(); i++)
		{
			int x = objects_rect[i].x;
			int y = objects_rect[i].y;
			int w = objects_rect[i].width;
			int h = objects_rect[i].height;
			int cx = (x + x + w) / 2;
			int cy = (y + y + h) / 2;

			//Find out if that object was detected already
			bool same_object_detected = false;
			for (map<int, Point>::iterator it = center_points.begin(); it != center_points.end(); it++)
			{
				int center_id = it->first;
				float dist_squre = hypot(cx - it->second.x, cy - it->second.y);
				if (dist_squre < 25)
				{
					center_points[center_id] = Point(cx, cy);
					cout << center_id << ":" << Point(cx, cy) << endl;
					objects_bbs_ids.insert(pair<int, Rect>(center_id, Rect(x, y, w, h)));
					same_object_detected = true;
					break;
				}
			}
			//New object is detected we assign the ID to that object
			if (same_object_detected == false)
			{
				center_points[id_count] = Point(cx, cy);
				objects_bbs_ids.insert(pair<int, Rect>(id_count, Rect(x, y, w, h)));
				id_count += 1;
			}
		}
		//Clean the dictionary by center points to remove IDS not used anymore
		map<int, Point>new_center_points;
		for (map<int, Rect>::iterator it2 = objects_bbs_ids.begin(); it2 != objects_bbs_ids.end(); it2++)
		{
			int old_id = it2->first;
			Point center = center_points[old_id];
			new_center_points[old_id] = center;
		}
		//Update dictionary with IDs not used removed
		center_points = new_center_points;

		return objects_bbs_ids;
	}
};


int inpWidth = 320;					//輸入維度
int inpHeight = 320;
string modelConfiguration = "E:\\opencv\\YOLO\\yolov3.cfg";
string weights = "E:\\opencv\\YOLO\\yolov3.weights";
vector<Rect> bboxes;

vector<Rect> findObjects(vector<Mat>& out, Mat& img, vector<string>classNames, vector<Rect>dections, float confThreshold = 0.35, float nmsThreshold = 0.3,string target="all")
{
	int wt, ht;
	wt = img.size().width,
	ht = img.size().height;

	vector<int> classIds;			// 存放檢測物體類別
	vector<float> confidences;// 檢測置信度
	vector<Rect> boxes;
	

	//out[0] [1] [2] 3個檢測頭  out[i].rows  第x個先驗框    [507,85]  [2028,85] ,[8112,85]

	//out[i].data :pointer ： 指向  存放85個值的容器 
	//遍歷3個檢測頭
	for (int i = 0; i < out.size(); i++)
	{

		float* data = (float*)out[i].data;
		// j:遍歷檢測頭的先驗框   data pointer: 每次加85，找到下一個先驗框 
		for (int j = 0; j < out[i].rows; j++, data += out[i].cols)
		{
			//80個概率 找最大
			Mat scores_ = out[i].row(j).colRange(5, out[i].cols);
			Point classIdPoint;
			double confidence;
			//找出最大概率值以及 對應 號碼
			minMaxLoc(scores_, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * wt);
				int centerY = (int)(data[1] * ht);
				int width = (int)(data[2] * wt);
				int height = (int)(data[3] * ht);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				if (target == "all")
				{
					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
				else
				{
					if (classNames[classIdPoint.x] == target)
					{
						classIds.push_back(classIdPoint.x);
						confidences.push_back((float)confidence);
						boxes.push_back(Rect(left, top, width, height));
					}
				}
				

			}
		}

		vector<int> index;
		dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, index);

		
		for (int jj = 0; jj < index.size(); jj++)
		{

			int sub_index = index[jj];
			bboxes.push_back(boxes[sub_index]);
			int x = boxes[sub_index].x;
			int y = boxes[sub_index].y;
			int w = boxes[sub_index].width;
			int h = boxes[sub_index].height;
			//rectangle(img, boxes[sub_index], Scalar(0, 0, 255), 1);
			Rect box = Rect(x, y, w, h);
			dections.push_back(box);
			//string label = format("%.2f", confidences[sub_index]);
			//label = classNames[classIds[sub_index]] + ":" + label;
			//putText(img, label, Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2);
		}
		

	}
	return dections;

}



void YOLOV3()
{
	//read class names
	vector<string>classNames;
	string classFile = "E:\\opencv\\YOLO\\coco.names";
	ifstream ifs;
	ifs.open(classFile, ios::in);
	string line;
	while (getline(ifs, line))
	{
		classNames.push_back(line);
	}

	VideoCapture capture("E:\\object_tracking\\object_tracking\\highway.mp4");
	Mat img, blob,roi;
	vector<string>outputnames;
	vector<Mat> outputs;
	dnn::Net net = dnn::readNetFromDarknet(modelConfiguration, weights);
	net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(dnn::DNN_TARGET_CPU);
	EuclideanDistTracker tracker;

	while (true)
	{
		capture.read(img);

		if (img.empty())
		{
			break;
		}
		roi = img(Range(340, 720), Range(500, 800));
		//preprocess
		blob = dnn::blobFromImage(roi, 1.0 / 255, Size(inpWidth, inpHeight));
		net.setInput(blob);
		vector<string> layerNames = net.getLayerNames();
		vector<int> allOutputLayers = net.getUnconnectedOutLayers();

		for (int ii = 0; ii < allOutputLayers.size(); ii++)
		{
			int index = (int)allOutputLayers[ii];
			outputnames.push_back(layerNames[index - 1]);
		}
		net.forward(outputs, outputnames);
		//1:DETECTION
		vector<Rect>dections;
		dections=findObjects(outputs, roi, classNames,dections,0.65,0.45,"all");
		//2:TRACKING
		map<int, Rect>boxes_ids = tracker.update(dections);
		for (map<int, Rect>::iterator it3 = boxes_ids.begin(); it3 != boxes_ids.end(); it3++)
		{
			int x = it3->second.x;
			int y = it3->second.y;
			int w = it3->second.width;
			int h = it3->second.height;
			string id = to_string(it3->first);
			rectangle(roi, Rect(x, y, w, h), Scalar(0, 0, 255), 2);
			putText(roi, id, Point(x, y - 15), FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2);
		}
		imshow("frame", img);
		imshow("roi", roi);
		int key =cv::waitKey(1);
		if (key == 27)
		{
			break;
		}
	}
}


const size_t inWidth = 300;
const size_t inHeight = 300;
int SSD()
{
	vector<string>classNames;
	string classFile = "E:\\opencv\\Object_Detection_SSD\\coco.names";
	ifstream ifs;
	ifs.open(classFile, ios::in);
	string line;
	while (getline(ifs, line))
	{
		classNames.push_back(line);
	}

	clock_t start, finish;
	double totaltime;
	Mat frame,roi;
	VideoCapture capture("E:\\object_tracking\\object_tracking\\highway.mp4");
	String weights = "E:\\opencv\\Object_Detection_SSD\\frozen_inference_graph.pb";
	String prototxt = "E:\\opencv\\Object_Detection_SSD\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
	dnn::Net net = cv::dnn::readNetFromTensorflow(weights, prototxt);
	EuclideanDistTracker tracker;

	while (capture.read(frame))
	{
		start = clock();
		//resize(frame, frame, Size(320, 320));
		roi = frame(Range(340, 720), Range(500, 800));
		int wt, ht;
		wt = roi.size().width;
		ht = roi.size().height;
		vector<Rect> boxes;


		cv::Mat blob = cv::dnn::blobFromImage(roi, 1. / 255, Size(inWidth, inHeight));
		//cout << "blob size: " << blob.size << endl;

		net.setInput(blob);
		Mat output = net.forward();
		//cout << "output size: " << output.size << endl;

		Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

		
		float confidenceThreshold = 0.4;
		vector<float>confidences;
		vector<int>classIds;

		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > confidenceThreshold)
			{
				size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
				classIds.push_back(objectClass-1);

				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * wt);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * ht);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * wt);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * ht);

				ostringstream ss;
				ss << confidence;
				String conf(ss.str());

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));
				confidences.push_back(confidence);
				boxes.push_back(object);
				//rectangle(roi, object, Scalar(0, 255, 0), 2);
				//String label = String(classNames[objectClass-1]) + ": " + conf;
				//putText(roi, label, Point(xLeftBottom, yLeftBottom-15),FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
		vector<int> RemainIndex;
		dnn::NMSBoxes(boxes, confidences, confidenceThreshold, 0.4, RemainIndex);
		vector<Rect>dections;

		for (int jj = 0; jj < RemainIndex.size(); jj++)
		{
			int sub_index = RemainIndex[jj];
			if (classNames[classIds[sub_index]] == "person")
			{
				
				bboxes.push_back(boxes[sub_index]);
				int x = boxes[sub_index].x;
				int y = boxes[sub_index].y;
				int w = boxes[sub_index].width;
				int h = boxes[sub_index].height;
				rectangle(roi, boxes[sub_index], Scalar(0, 0, 255), 1);
				Rect box = Rect(x, y, w, h);
				dections.push_back(box);
				/*string label = format("%.2f", confidences[sub_index]);
				label = classNames[classIds[sub_index]] + ":" + label;
				putText(roi, label, Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2);*/
			}
		}
		map<int, Rect>boxes_ids = tracker.update(dections);
		for (map<int, Rect>::iterator it3 = boxes_ids.begin(); it3 != boxes_ids.end(); it3++)
		{
			int x = it3->second.x;
			int y = it3->second.y;
			int w = it3->second.width;
			int h = it3->second.height;
			string id = to_string(it3->first);
			rectangle(roi, Rect(x, y, w, h), Scalar(0, 0, 255), 2);
			putText(roi, id, Point(x, y - 15), FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2);
		}


		finish = clock();
		totaltime = finish - start;
		cout << "TIME:" << totaltime << "ms" << endl;
		
		imshow("result", frame);
		imshow("roi", roi);
		char c = cv::waitKey(30);
		if (c == 27)
		{ // ESC退出
			break;
		}
	}
	capture.release();
	cv::waitKey(30);
	return 0;
}


void main()
{
	//YOLOV3();
	SSD();
}
