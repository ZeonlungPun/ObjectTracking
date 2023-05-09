#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <opencv2\opencv.hpp>
#include "opencv2/core.hpp"
#include<iostream>
#include <algorithm>
#include<vector>
#include<string>
#include<fstream>
#include<math.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

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
		for (int i=0;i<objects_rect.size();i++)
		{
			int x = objects_rect[i].x;
			int y = objects_rect[i].y;
			int w = objects_rect[i].width;
			int h = objects_rect[i].height;
			int cx = (x + x + w) /2;
			int cy = (y + y + h) /2;
			
			//Find out if that object was detected already
			bool same_object_detected = false;
			for (map<int,Point>::iterator it=center_points.begin();it!=center_points.end();it++)
			{
				int center_id =it->first;
				float dist_squre = hypot(cx - it->second.x, cy - it->second.y); 
				if (dist_squre < 25)
				{
					center_points[center_id] = Point(cx, cy);
					cout << center_id << ":" << Point(cx, cy) << endl;
					objects_bbs_ids.insert(pair<int,Rect>(center_id, Rect(x, y, w, h)));
					same_object_detected = true;
					break;
				}
			}
			//New object is detected we assign the ID to that object
			if (same_object_detected == false)
			{
				center_points[id_count]= Point(cx, cy);
				objects_bbs_ids.insert(pair<int, Rect>(id_count, Rect(x, y, w, h)));
				id_count += 1;
			}
		}
		//Clean the dictionary by center points to remove IDS not used anymore
		map<int, Point>new_center_points;
		for (map<int,Rect>::iterator it2=objects_bbs_ids.begin();it2!=objects_bbs_ids.end();it2++)
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


void main()
{
	//creater tracker
	EuclideanDistTracker tracker;
	//read vedio
	VideoCapture capture("E:\\object_tracking\\object_tracking\\highway.mp4");
	//Object detection from Stable camera
	Ptr<BackgroundSubtractorMOG2>MOG = createBackgroundSubtractorMOG2(100,40);
	Mat frame,roi,mask;
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	while (true)
	{
		capture.read(frame);
		if (frame.empty())
		{
			break;
		}
		roi = frame(Range(340,720),Range(500,800));
		//1. Object Detection
		(*MOG).apply(roi, mask);
		threshold(mask, mask, 254, 255, THRESH_BINARY);
		findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
		vector<Rect>dections;
		for (int j = 0; j < contours.size(); j++)
		{
			int area = contourArea(contours[j]);
			float peri = arcLength(contours[j], true);
			vector<vector<Point>>conPoly(contours.size());
			approxPolyDP(contours[j], conPoly[j], 0.02 * peri, true);

			if (area > 100 )
			{
				RotatedRect rect = minAreaRect(contours[j]);
				Rect box = rect.boundingRect();
				dections.push_back(box);
			}
		}
		//2. Object Tracking
		map<int, Rect>boxes_ids = tracker.update(dections);
		for (map<int, Rect>::iterator it3 = boxes_ids.begin(); it3 != boxes_ids.end(); it3++)
		{
			int x = it3->second.x;
			int y = it3->second.y;
			int w = it3->second.width;
			int h = it3->second.height;
			string id =to_string(it3->first);
			rectangle(roi, Rect(x, y,w, h), Scalar(0, 0, 255), 2);
			putText(roi, id, Point(x, y - 15), FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2);
		}
		imshow("frame", roi);
		imshow("mask",mask);
		int key = waitKey(10);
		if (key == 27)
		{
			break;
		}
	}
	
}
