#ifndef SIMPLEVO_OPTIMIZATION_H
#define SIMPLEVO_OPTIMIZATION_H

#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <unordered_map>

namespace SimpleVO
{
    class Optimize
    {
    public:
        void SetIntrinsic(double fx, double fy, double cx, double cy);
        void AddParameters(Sophus::SE3d& pose);
        void AddOvservation(double px, double py,
            Eigen::Matrix<double, 3, 1>& p3d,
            Sophus::SE3d& pose);
        void SetOptions();
        void Run();

    public:
        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        double fx, fy, cx, cy;
    };
}
#endif // SIMPLEVO_OPTIMIZATION_H