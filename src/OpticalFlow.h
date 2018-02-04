#ifndef SIMPLEVO_OPTICAL_FLOW_H
#define SIMPLEVO_OPTICAL_FLOW_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace SimpleVO
{

    /**
     * single level optical flow
     * @param [in] img1 the first image
     * @param [in] img2 the second image
     * @param [in] kp1 keypoints in img1
     * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
     * @param [out] success true if a keypoint is tracked successfully
     * @param [in] inverse use inverse formulation?
     */
    void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<cv::KeyPoint> &kp1,
        std::vector<cv::KeyPoint> &kp2,
        std::vector<bool> &success,
        bool inverse = false
    );

    /**
     * multi level optical flow, scale of pyramid is set to 2 by default
     * the image pyramid will be create inside the function
     * @param [in] img1 the first pyramid
     * @param [in] img2 the second pyramid
     * @param [in] kp1 keypoints in img1
     * @param [out] kp2 keypoints in img2
     * @param [out] success true if a keypoint is tracked successfully
     * @param [in] inverse set true to enable inverse formulation
     */
    void OpticalFlowMultiLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<cv::KeyPoint> &kp1,
        std::vector<cv::KeyPoint> &kp2,
        std::vector<bool> &success,
        bool inverse = false
    );
}

#endif // SIMPLEVO_OPTICAL_FLOW_H