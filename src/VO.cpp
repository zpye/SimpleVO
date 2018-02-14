#include "VO.h"
#include "OpticalFlow.h"
#include "direct.h"
#include "optimization.h"
#include "utils.hpp"

using namespace std;
using namespace cv;

namespace SimpleVO
{
    VO::VO(double _fx, double _fy, double _cx, double _cy, double _baseline)
        : fx(_fx), fy(_fy), cx(_cx), cy(_cy), baseline(_baseline)
    {
        isInitialized = false;
        frameCounter = 0;
        mapPointsNum = 0;
        keyFramesNum = 0;
        lastFrame = nullptr;
        thisFrame = nullptr;
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

    void VO::CreateFrameByStereoImage(const Mat& left, const Mat& right,
        bool provideLeftKeyPoints, vector<KeyPoint>& kp_left, 
        vector<Point3d>& p3d, vector<bool>& success)
    {
        if(!p3d.empty())
        {
            p3d.clear();
        }

        if(!success.empty())
        {
            success.clear();
        }

        if(!provideLeftKeyPoints)
        {
            if(!kp_left.empty())
            {
                kp_left.clear();
            }

            // key points, using GFTT here.
            Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
            detector->detect(left, kp_left);
            cout << "detect num: " << kp_left.size() << endl;
        }
       
        // get corresponding points in the other image
        vector<KeyPoint> kp_right;
        OpticalFlowMultiLevel(left, right,
            kp_left, kp_right, success, true);

        // get depth of key points
        Get3DPoint(kp_left, kp_right, success, p3d);
    }

    bool VO::IsKeyFrame(const Frame* const f)
    {
        // number of key points too small
        unsigned int points_num = f->points.size();
        if(points_num < 50)
        {
            cout << "keypoints number: " << points_num << endl;
            return true;
        }

        return false;
    }

    void VO::Init(const Mat& left, const Mat& right)
    {
        thisLeftImg = left;
        thisRightImg = right;

        // initialize frame
        vector<KeyPoint> kp_left;
        vector<Point3d> points3d;
        vector<bool> success;

        CreateFrameByStereoImage(thisLeftImg, thisRightImg,
            false, kp_left, points3d, success);

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
            mapPoints[mapPointsNum] = p3d;
            mapPointsNum += 1;
        }

        // add frame to map
        // first frame is key frame
        f->isKeyFrame = true;
        keyFrames[keyFramesNum] = f;
        keyFramesNum += 1;
        
        thisFrame = f;
    }

    void VO::AddStereoImage(const Mat& left, const Mat& right)
    {
        frameCounter += 1;

        // initialization, first frame
        if(!isInitialized)
        {
            Init(left, right);
            isInitialized = true;
            return;
        }

        // save last frame and images
        if(lastFrame != nullptr)
        {
            if(!lastFrame->isKeyFrame)
            {
                delete lastFrame;
            }
        }
        lastFrame = thisFrame;
        lastLeftImg = thisLeftImg;
        lastRightImg = thisRightImg;

        Frame* f = new Frame;
        thisLeftImg = left;
        thisRightImg = right;

        // get depth for this frame
        vector<KeyPoint>& kp_left = lastFrame->points;
        vector<Point3d> points3d;
        vector<bool> success; 

        CreateFrameByStereoImage(thisLeftImg, thisRightImg,
            true, kp_left, points3d, success);

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
        
        // add KeyFrame
        if(IsKeyFrame(f))
        {
            // add frame to map
            f->isKeyFrame = true;
            keyFrames[keyFramesNum] = f;
            keyFramesNum += 1;

            // add key frame reference to points3d
            for(unsigned int i = 0; i < f->IDs.size(); ++i)
            {
                Point3d* p3d = mapPoints[f->IDs[i]];
                p3d->observedKeyFrames[p3d->observedKeyFramesNum] = keyFramesNum - 1;
                p3d->observedKeyFramesNum += 1;
            }

            // optimize
            Optimize opt;
            opt.SetIntrinsic(fx, fy, cx, cy);

            // use the nearest 5 key frames to optimize
            unsigned int lower_bound = 0;
            if(keyFramesNum > 5)
            {
                lower_bound = keyFramesNum - 5;
            }
            
            for(unsigned int i = lower_bound; i < keyFramesNum; ++i)
            {
                Frame* f = keyFrames[i];
                opt.AddParameters(f->pose);
                for(unsigned int j = 0; j < f->IDs.size(); ++j)
                {
                    opt.AddOvservation(f->points[j].pt.x, f->points[j].pt.y,
                        mapPoints[f->IDs[j]]->p, f->pose);
                }
            }

            opt.SetOptions();
            opt.Run();

            // detect new key points
            vector<KeyPoint> kp_new;
            vector<Point3d> p3d_new;
            vector<bool> success_new;
            CreateFrameByStereoImage(thisLeftImg, thisRightImg,
                false, kp_new, p3d_new, success_new);

            // remove duplicate key points
            RemoveDuplicateKeyPoints(kp_new, f->points, success_new);

            // add new points
            for(unsigned int i = 0; i < success_new.size(); ++i)
            {
                if(!success_new[i])
                    continue;

                // add 3d point
                Point3d* p3d = new Point3d(p3d_new[i]);
                p3d->observedKeyFrames[p3d->observedKeyFramesNum] = keyFramesNum - 1;
                p3d->observedKeyFramesNum += 1;

                // add 2d points to frame
                f->addPoint(kp_new[i], mapPointsNum);

                // add 3d points to map
                p3d->p = f->pose.inverse() * p3d->p;
                mapPoints[mapPointsNum] = p3d;
                mapPointsNum += 1;
            }
        }
        
        thisFrame = f;
    }

    void VO::PrintStatus()
    {
        // print status of current frame
        cout << "============================================" << endl;
        cout << "Frame number: " << frameCounter << endl;
        cout << "map points number: " << mapPointsNum << endl;
        cout << "key frames number: " << keyFramesNum << endl;
        cout << "current pose:\n" << thisFrame->pose.inverse().matrix() << endl;
    }

    void VO::WriteToFile(ofstream& out)
    {
        Eigen::Matrix<double, 3, 4> pose = thisFrame->pose.inverse().matrix3x4();
        out << pose(0, 0) << " ";
        out << pose(0, 1) << " ";
        out << pose(0, 2) << " ";
        out << pose(0, 3) << " ";
        out << pose(1, 0) << " ";
        out << pose(1, 1) << " ";
        out << pose(1, 2) << " ";
        out << pose(1, 3) << " ";
        out << pose(2, 0) << " ";
        out << pose(2, 1) << " ";
        out << pose(2, 2) << " ";
        out << pose(2, 3) << " ";
        out << endl;
    }
}