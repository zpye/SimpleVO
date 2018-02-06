#ifndef SIMPLEVO_FRAME_H
#define SIMPLEVO_FRAME_H

#include "MapPoint.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace SimpleVO
{

    class Frame
    {
    public:
        Frame() : isKeyFrame(false)
        {
        }

        void addPoint(cv::KeyPoint p, unsigned int id)
        {
            points.push_back(p);
            IDs.push_back(id);
        }

    public:
        Sophus::SE3d pose;
        std::vector<cv::KeyPoint> points;
        std::vector<unsigned int> IDs;
        bool isKeyFrame;
    };
}

#endif // SIMPLEVO_FRAME_H