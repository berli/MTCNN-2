#include <iostream>
#include "MTCNN.h"
#include "cv.h"
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

int main(int argc, char*argv[]) 
{
    //the vector used to input the address of the net model
    vector<string> model_file = {
            "model/det1.prototxt",
            "model/det2.prototxt",
            "model/det3.prototxt"
//          "../model/det4.prototxt"
    };

    //the vector used to input the address of the net parameters
    vector<string> trained_file = {
            "model/det1.caffemodel",
            "model/det2.caffemodel",
            "model/det3.caffemodel"
//            "../model/det4.caffemodel"
    };

    MTCNN mtcnn;
    mtcnn.initialize(model_file, trained_file);
    
    Mat img;
    VideoCapture cap;
    bool lbRet = cap.open("../../hls-20-720p.mp4");
    if(!lbRet)
       lbRet = cap.open(0);

    if(argc ==2)
       img = imread(argv[1]);

//    VideoWriter writer;
//    writer.open("../result/SuicideSquad.mp4",CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280,720), true);

    unsigned long  frame_count = 0;
    while(true)
    {
        if(lbRet)
	      lbRet = cap.read(img);
        vector<Rect> rectangles;
        vector<float> confidences;
        std::vector<std::vector<cv::Point>> alignment;
        
        long s = getMillSeconds();
        mtcnn.detection(img, rectangles, confidences, alignment);
        
		int liFps = getMillSeconds() - s;
        if( liFps > 0 )
            liFps = 1000/liFps;
        else
            liFps = 1000;
        
        string lsFps = "fps:";
        lsFps += to_string(liFps);
        lsFps += " time is:";
        lsFps += to_string(getMillSeconds()-s);
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
        if(!lbRet)
           LOG(INFO)<<"face num:"<<rectangles.size()<<" alignment:"<<alignment.size();

        frame_count++;
        cv::putText(img, lsFps, cvPoint(3, 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
//        writer.write(img);
        imshow("Live", img);
        if(!lbRet)
		{
         waitKey(0);
	     break;
		}
	    else
		{
         waitKey(1);
		}
    }

    return 0;
}


