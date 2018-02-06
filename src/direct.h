#ifndef SIMPLEVO_DIRECT_H
#define SIMPLEVO_DIRECT_H

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <vector>

#include "utils.hpp"

namespace SimpleVO
{
    // useful typedefs
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;
    typedef Eigen::Matrix<double, 2, 6> Matrix26d;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;

    /**
    * pose estimation using direct method
    * @param img1
    * @param img2
    * @param px_ref
    * @param depth_ref
    * @param T21
    */
    void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const std::vector<double>& depth_ref,
        VecVector2d& goodProjection,
        std::vector<unsigned int>& index,
        Sophus::SE3d &T21,
        double fx, double fy, double cx, double cy
    );

    /**
    * pose estimation using direct method
    * @param img1
    * @param img2
    * @param px_ref
    * @param depth_ref
    * @param T21
    */
    void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const std::vector<double>& depth_ref,
        VecVector2d& goodProjection,
        std::vector<unsigned int>& index,
        Sophus::SE3d &T21,
        double fx, double fy, double cx, double cy
    );
}

#endif // SIMPLEVO_DIRECT_H