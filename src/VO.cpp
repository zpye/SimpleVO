#include "VO.h"
#include "OpticalFlow.h"

using namespace std;
using namespace cv;

namespace SimpleVO
{
    VO::VO()
    {
        isInitialized = false;
        mapPointsNum = 0;
        keyFramesNum = 0;
    }

    void VO::Init(const Mat& left, const Mat& right)
    {
        thisLeftImg = left;
        thisRightImg = right;

        // initialize frame
        // key points, using GFTT here.
        vector<KeyPoint> kp1;
        Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
        detector->detect(thisLeftImg, kp1);

        // get corresponding points in the other image
        vector<KeyPoint> kp2;
        vector<bool> success;
        OpticalFlowMultiLevel(thisLeftImg, thisRightImg,
            kp1, kp2, success, true);
    }

    void VO::AddStereoImage(const Mat& left, const Mat& right)
    {
        // initialization
        if(!isInitialized)
        {
            Init(left, right);
            return;
        }
    }
}