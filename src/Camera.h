#ifndef SIMPLEVO_CAMERA_H
#define SIMPLEVO_CAMERA_H

#include <sophus/se3.hpp>

class Camera
{
public:
    Camera() {}
    ~Camera() {}

    Camera(double _fx, double _fy, double _cx, double _cy,
        Sophus::SE3d _pose)
        : fx(_fx), fy(_fy), cx(_cx), cy(_cy), pose(_pose)
    {
    }

    Camera(const Camera& cam) : fx(cam.fx), fy(cam.fy), cx(cam.cx), 
        cy(cam.cy), pose(cam.pose)
    {
    }

public:
    double fx;
    double fy;
    double cx;
    double cy;
    Sophus::SE3d pose;
};

#endif // SIMPLEVO_CAMERA_H