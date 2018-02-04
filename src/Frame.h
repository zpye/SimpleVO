#ifndef SIMPLEVO_FRAME_H
#define SIMPLEVO_FRAME_H

#include "Camera.h"
#include "MapPoint.h"
#include <vector>

namespace SimpleVO
{

    class Frame
    {
    public:
        Frame() {}
        ~Frame() {}

        Frame(Camera cam)
            : camera(cam), isKeyFrame(false)
        {
        }

        void addPoint(Point2d p2d)
        {
            points.push_back(p2d);
        }

    public:
        Camera camera;
        std::vector<Point2d> points;
        bool isKeyFrame;
    };
}

#endif // SIMPLEVO_FRAME_H