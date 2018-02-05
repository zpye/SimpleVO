#include "VO.h"
#include "OpticalFlow.h"
#include "utils.hpp"

using namespace std;
using namespace cv;

namespace SimpleVO
{
    VO::VO(double _fx, double _fy, double _cx, double _cy, double _baseline)
        : fx(_fx), fy(_fy), cx(_cx), cy(_cy), baseline(_baseline)
    {
        isInitialized = false;
        mapPointsNum = 0;
        keyFramesNum = 0;
    }

    void VO::Get3DPoint(const vector<KeyPoint>& kp_left, 
        const vector<KeyPoint>& kp_right,
        vector<bool>& success,
        vector<Point3d>& points3d)
    {
        if(!points3d.empty())
        {
            points3d.clear();
        }

        // get disparity
        vector<double> disparity;
        ComputeDisparity(kp_left, kp_right, success, disparity);

        for(unsigned int i = 0; i < success.size(); ++i)
        {
            if(!success[i])
            {
                points3d.push_back(Point3d());
                continue;
            }

            double disp = disparity[i];
            double Z = fx * baseline / disp;
            double Xn = (kp_left[i].pt.x - cx) / fx;
            double Yn = (kp_left[i].pt.y - cy) / fy;
            points3d.push_back(Point3d(Xn * Z, Yn * Z, Z));
        }
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

        // get depth of key points
        vector<Point3d> points3d;
        Get3DPoint(kp1, kp2, success, points3d);

        // initial frame and map
        Frame* f = new Frame;
        for(unsigned int i = 0; i < success.size(); ++i)
        {
            if(!success[i])
                continue;

            // add 3d point
            Point3d* p3d = new Point3d(points3d[i]);
            p3d->observedKeyFrames[p3d->observedKeyFramesNum] = keyFramesNum;
            p3d->observedKeyFramesNum += 1;
            
            // add 2d points to frame
            f->addPoint(Point2d(kp1[i].pt.x, kp1[i].pt.y, mapPointsNum));

            // add 3d points to map
            mapPoints[mapPointsNum] = p3d;
            mapPointsNum += 1;
        }

        // add frame to map
        // first frame is key frame
        keyFrames[keyFramesNum] = f;
        keyFramesNum += 1;
    }

    void VO::AddStereoImage(const Mat& left, const Mat& right)
    {
        // initialization
        if(!isInitialized)
        {
            Init(left, right);
            isInitialized = true;
            return;
        }


    }
}