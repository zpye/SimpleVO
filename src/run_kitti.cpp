#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <boost/format.hpp>

#include "VO.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) 
{
	boost::format leftImages("E:/SLAM_class/project/SimpleVO/data/00/image_0/%06d.png");
	boost::format rightImages("E:/SLAM_class/project/SimpleVO/data/00/image_1/%06d.png");

    // initialize
    // Camera intrinsics
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    double baseline = 0.573;

    SimpleVO::VO* myVO = new SimpleVO::VO(fx, fy, cx, cy, baseline);

    // add images
    unsigned int ImageNum = 1000;
    for(unsigned int i = 0; i < ImageNum; ++i)
    {
        Mat leftImg = imread((leftImages % i).str(), 0);
        Mat rightImg = imread((rightImages % i).str(), 0);

        myVO->AddStereoImage(leftImg, rightImg);    
        
        // print info
        myVO->printStatus();
    }

    cout << "END" << endl;
	return 0;
}