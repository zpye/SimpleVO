#ifndef SIMPLEVO_MAP_POINT_H
#define SIMPLEVO_MAP_POINT_H

#include <Eigen/Eigen>

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
    unsigned int id;
};

class Point3d
{
public:
    Point3d() {}
    ~Point3d() {}

    Point3d(double _x, double _y, double _z)
        : p(_x, _y, _z)
    {
    }

public:
    Eigen::Matrix<double, 3, 1> p;
};

#endif // SIMPLEVO_MAP_POINT_H