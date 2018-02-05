#ifndef SIMPLEVO_FRAME_H
#define SIMPLEVO_FRAME_H

#include "MapPoint.h"
#include <vector>

namespace SimpleVO
{

    class Frame
    {
    public:
        Frame() : isKeyFrame(false)
        {
        }

        void addPoint(Point2d p2d)
        {
            points.push_back(p2d);
        }

    public:
        Sophus::SE3d pose;
        std::vector<Point2d> points;
        bool isKeyFrame;
    };
}

#endif // SIMPLEVO_FRAME_H