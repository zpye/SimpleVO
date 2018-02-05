#ifndef SIMPLEVO_MAP_POINT_H
#define SIMPLEVO_MAP_POINT_H

#include <Eigen/Eigen>
#include <unordered_map>

namespace SimpleVO
{
    class Point2d
    {
    public:
        Point2d() {}
        ~Point2d() {}

        Point2d(double _x, double _y, unsigned int _id)
            : p(_x, _y), id(_id)
        {
        }

    public:
        Eigen::Matrix<double, 2, 1> p;
        Eigen::Matrix<double, 16, 1> color;
        unsigned int id; // id of 3d point
    };

    class Point3d
    {
    public:
        Point3d() {}
        ~Point3d() {}

        Point3d(double _x, double _y, double _z)
            : p(_x, _y, _z), observedKeyFramesNum(0)
        {
        }

        Point3d(Eigen::Matrix<double, 3, 1>& _p)
            : p(_p), observedKeyFramesNum(0)
        {
        }

        Point3d(const Point3d& _p)
            : p(_p.p), observedKeyFramesNum(0)
        {
        }

    public:
        Eigen::Matrix<double, 3, 1> p;
        std::unordered_map<unsigned int, unsigned int> observedKeyFrames;
        unsigned int observedKeyFramesNum;
    };
}

#endif // SIMPLEVO_MAP_POINT_H