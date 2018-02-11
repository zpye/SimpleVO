#ifndef SIMPLEVO_UTILS_HPP
#define SIMPLEVO_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>

namespace SimpleVO
{
    typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

    /**
    * get a gray scale value from reference image (bi-linear interpolated)
    * @param img
    * @param x
    * @param y
    * @return
    */
    inline float GetPixelValue(const cv::Mat &img, float x, float y) {
        uchar *data = &img.data[int(y) * img.step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
            );
    }

    void ConvertKeyPointToEigen(const std::vector<cv::KeyPoint>& kp, 
        VecVector2d& vec2d);

    void ConvertEigenToKeyPoint(const VecVector2d& vec2d, 
        std::vector<cv::KeyPoint>& kp);

    void ComputeDisparity(const std::vector<cv::KeyPoint>& kp1, 
        const std::vector<cv::KeyPoint>& kp2,
        std::vector<bool>& success,
        std::vector<double>& disparity);

    void RemoveDuplicateKeyPoints(const std::vector<cv::KeyPoint>& kp_in,
        const std::vector<cv::KeyPoint>& kp_ref, std::vector<bool>& success);
}
#endif // SIMPLEVO_UTILS_HPP