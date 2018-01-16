#include <iostream>
#include "MTCNN.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//int main() {
//
//    vector<string> model_file = {
//            "../model/det1.prototxt",
//            "../model/det2.prototxt",
//            "../model/det3.prototxt"
////            "../model/det4.prototxt"
//    };
//
//    vector<string> trained_file = {
//            "../model/det1.caffemodel",
//            "../model/det2.caffemodel",
//            "../model/det3.caffemodel"
////            "../model/det4.caffemodel"
//    };
//
//    MTCNN mtcnn(model_file, trained_file);
//
//    vector<Rect> rectangles;
//    string img_path = "../result/trump.jpg";
//    Mat img = imread(img_path);
//
//    mtcnn.detection(img, rectangles);
//
//    std::cout << "Hello, World!" << std::endl;
//    return 0;
//}

long getMillSeconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return 1000*tv.tv_sec+tv.tv_usec/1000;
}

int main(int argc, char*argv[]) {


    //the vector used to input the address of the net model
    vector<string> model_file = 
	{
            "../model/det1.prototxt",
            "../model/det2.prototxt",
            "../model/det3.prototxt"
//            "../model/det4.prototxt"
    };

    //the vector used to input the address of the net parameters
    vector<string> trained_file = 
	{
            "../model/det1.caffemodel",
            "../model/det2.caffemodel",
            "../model/det3.caffemodel"
//            "../model/det4.caffemodel"
    };

    MTCNN mtcnn(model_file, trained_file);

	LOG(INFO)<<"argc:"<<argc;
    //VideoCapture cap("../../hls-20.mp4");
    VideoCapture cap;
	bool lbret = false;
	if(argc > 1)
		lbret = cap.open(argv[1]);
	else
		lbret = cap.open(0);

	if(!lbret)
		lbret = cap.open("../../hls-20.mp4");
	if(!lbret)
	{
		LOG(ERROR)<<"open failed";
		return 0;
	}
//    VideoCapture cap(0);

//    VideoWriter writer;
//    writer.open("../result/SuicideSquad.mp4",CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280,720), true);

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    Mat img;
    unsigned long  frame_count = 0;
    while(cap.read(img))
    {
        vector<Rect> rectangles;
        vector<float> confidences;
        std::vector<std::vector<cv::Point>> alignment;

        long s = getMillSeconds();
        mtcnn.detection(img, rectangles, confidences, alignment);

        int liElapse = getMillSeconds()-s;

        //cout<<" time is  "<<liElapse<<" ms"<<endl;
        int liSpeed = 0;
        if( liElapse > 0 )
            liSpeed = 1000/liElapse;
        else
            liSpeed = 1000;

        string lsFps = "fps:";
        lsFps += to_string(liSpeed);
        lsFps += " time:";
        lsFps += to_string(liElapse);
        lsFps += " total fps:";
        lsFps += std::to_string(frame_count);

        for(int i = 0; i < rectangles.size(); i++)
        {
            int green = confidences[i] * 255;
            int red = (1 - confidences[i]) * 255;
            rectangle(img, rectangles[i], cv::Scalar(0, green, red), 3);
            for(int j = 0; j < alignment[i].size(); j++)
            {
                cv::circle(img, alignment[i][j], 5, cv::Scalar(255, 255, 0), 3);
            }
        }

        frame_count++;
        cv::putText(img, lsFps, cvPoint(3, 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
//        writer.write(img);
        imshow("Live", img);
        waitKey(1);
    }

    return 0;
}


