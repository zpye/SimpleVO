#include "VO.h"
#include "OpticalFlow.h"
#include "direct.h"
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
        vector<KeyPoint> kp_left;
        Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
        detector->detect(thisLeftImg, kp_left);

        // get corresponding points in the other image
        vector<KeyPoint> kp_right;
        vector<bool> success;
        OpticalFlowMultiLevel(thisLeftImg, thisRightImg,
            kp_left, kp_right, success, true);

        // get depth of key points
        vector<Point3d> points3d;
        Get3DPoint(kp_left, kp_right, success, points3d);

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
            f->addPoint(kp_left[i], mapPointsNum);

            // add 3d points to map
            f->isKeyFrame = true;
            mapPoints[mapPointsNum] = p3d;
            mapPointsNum += 1;
        }

        thisFrame = f;

        // add frame to map
        // first frame is key frame
        keyFrames[keyFramesNum] = f;
        keyFramesNum += 1;
    }

    void VO::AddStereoImage(const Mat& left, const Mat& right)
    {
        // initialization, first frame
        if(!isInitialized)
        {
            Init(left, right);
            isInitialized = true;
            return;
        }

        // save last frame and images
        lastFrame = thisFrame;
        lastLeftImg = thisLeftImg;
        lastRightImg = thisRightImg;

        Frame* f = new Frame;
        thisLeftImg = left;
        thisRightImg = right;

        // get depth for this frame
        vector<KeyPoint>& kp_left = lastFrame->points;

        // get corresponding points in the other image
        vector<KeyPoint> kp_right;
        vector<bool> success;
        OpticalFlowMultiLevel(thisLeftImg, thisRightImg,
            kp_left, kp_right, success, true);

        // get depth of key points
        vector<Point3d> points3d;
        Get3DPoint(kp_left, kp_right, success, points3d);

        vector<double> depth;
        for(unsigned int i = 0; i < success.size(); ++i)
        {
            if(success[i])
            {
                depth.push_back(points3d[i].p(2));
            }
            else
            {
                depth.push_back(-1.0);
            }
        }

        // direct method
        VecVector2d px_ref, goodProjection;
        vector<unsigned int> index;
        ConvertKeyPointToEigen(kp_left, px_ref);
        DirectPoseEstimationMultiLayer(lastLeftImg, thisLeftImg,
            px_ref, depth, goodProjection, index, f->pose,
            fx, fy, cx, cy);

        // construct frame
        f->pose = f->pose * lastFrame->pose;
        vector<KeyPoint> goodKeyPoints;
        ConvertEigenToKeyPoint(goodProjection, goodKeyPoints);
        for(unsigned int i = 0; i < index.size(); ++i)
        {
            f->addPoint(goodKeyPoints[i], lastFrame->IDs[index[i]]);
        }
        thisFrame = f;

        // condition for add KeyFrame
        cout << index.size() << endl;
    }
}