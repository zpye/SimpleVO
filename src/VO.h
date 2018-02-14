#ifndef SIMPLEVO_VO_H
#define SIMPLEVO_VO_H

#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "MapPoint.h"
#include "Frame.h"

namespace SimpleVO
{
    /*
      core of visual odometry
    */
    class VO
    {
    public:
        VO(double fx, double fy, double cx, double cy, double baseline);
        void Init(const cv::Mat& left, const cv::Mat& right);
        void AddStereoImage(const cv::Mat& left, const cv::Mat& right);
        void Get3DPoint(const std::vector<cv::KeyPoint>& kp_left,
            const std::vector<cv::KeyPoint>& kp_right,
            std::vector<bool>& success,
            std::vector<Point3d>& points3d);
        void CreateFrameByStereoImage(const cv::Mat& left, const cv::Mat& right,
            bool provideLeftKeyPoints, std::vector<cv::KeyPoint>& kp, 
            std::vector<Point3d>& p3d, std::vector<bool>& success);
        bool IsKeyFrame(const Frame* const f);
        void PrintStatus();
        void WriteToFile(std::ofstream& out);

    public:
        bool isInitialized;

        Frame* lastFrame;
        Frame* thisFrame;

        cv::Mat lastLeftImg, lastRightImg;
        cv::Mat thisLeftImg, thisRightImg;

        unsigned int frameCounter;

        std::unordered_map<unsigned int, Point3d *> mapPoints;
        std::unordered_map<unsigned int, Frame *> keyFrames;
        unsigned int mapPointsNum;
        unsigned int keyFramesNum;

        double fx, fy, cx, cy;
        double baseline;
    };
}

#endif // SIMPLEVO_VO_H