#include "direct.h"

using namespace std;
using namespace cv;

namespace SimpleVO
{
    void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21,
        double fx, double fy, double cx, double cy)
    {

        // parameters
        int half_patch_size = 4;
        int iterations = 100;

        double cost = 0, lastCost = 0;
        int nGood = 0;  // good projections
        VecVector2d goodProjection;

        for(int iter = 0; iter < iterations; iter++) {
            nGood = 0;
            goodProjection.clear();

            // Define Hessian and bias
            Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
            Vector6d b = Vector6d::Zero();  // 6x1 bias

            for(size_t i = 0; i < px_ref.size(); i++) {

                // compute the projection in the second image
                Eigen::Vector2d p1 = px_ref[i];
                Eigen::Vector3d P1((p1[0] - cx) / fx, (p1[1] - cy) / fy, 1);
                P1 *= depth_ref[i];
                Eigen::Vector3d P2 = T21 * P1;
                double X = P2[0];
                double Y = P2[1];
                double invz = 1.0 / P2[2];
                double invz2 = invz * invz;

                float u = 0, v = 0;
                u = X * fx * invz + cx;
                v = Y * fy * invz + cy;

                if(u <= half_patch_size || u >= img2.cols - half_patch_size ||
                    v <= half_patch_size || v >= img2.rows - half_patch_size) {
                    // go outside
                    continue;
                }

                nGood++;
                goodProjection.push_back(Eigen::Vector2d(u, v));

                // and compute error and jacobian
                for(int x = -half_patch_size; x < half_patch_size; x++)
                    for(int y = -half_patch_size; y < half_patch_size; y++) {
                        double error = GetPixelValue(img1, p1[0] + x, p1[1] + y) - GetPixelValue(img2, u + x, v + y);

                        Matrix26d J_pixel_xi;   // pixel to \xi in Lie algebra
                        Eigen::Vector2d J_img_pixel;    // image gradients

                        J_pixel_xi(0, 0) = fx * invz;
                        J_pixel_xi(0, 1) = 0;
                        J_pixel_xi(0, 2) = -fx * X * invz2;
                        J_pixel_xi(0, 3) = -fx * X * Y * invz2;
                        J_pixel_xi(0, 4) = fx * (1 + X * X * invz2);
                        J_pixel_xi(0, 5) = -fx * Y * invz;

                        J_pixel_xi(1, 0) = 0;
                        J_pixel_xi(1, 1) = fy * invz;
                        J_pixel_xi(1, 2) = -fy * Y * invz2;
                        J_pixel_xi(1, 3) = -fy * (1 + Y * Y * invz2);
                        J_pixel_xi(1, 4) = fy * X * Y * invz2;
                        J_pixel_xi(1, 5) = fy * X * invz2;

                        J_img_pixel(0) = 0.5 * (GetPixelValue(img2, u + x + 1, v + y) - GetPixelValue(img2, u + x - 1, v + y));
                        J_img_pixel(1) = 0.5 * (GetPixelValue(img2, u + x, v + y + 1) - GetPixelValue(img2, u + x, v + y - 1));

                        // total jacobian
                        Vector6d J = -J_pixel_xi.transpose() * J_img_pixel;

                        H += J * J.transpose();
                        b += -error * J;
                        cost += error * error;
                    }
            }

            // solve update and put it into estimation
            Vector6d update = H.ldlt().solve(b);
            T21 = Sophus::SE3d::exp(update) * T21;

            cost /= nGood;

            if(isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                break;
            }
            if(iter > 0 && cost > lastCost) {
                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }
            lastCost = cost;
            cout << "cost = " << cost << ", good = " << nGood << endl;
        }
        cout << "good projection: " << nGood << endl;
        cout << "T21 = \n" << T21.matrix() << endl;

        // in order to help you debug, we plot the projected pixels here
        cv::Mat img1_show, img2_show;
        cv::cvtColor(img1, img1_show, CV_GRAY2BGR);
        cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
        for(auto &px : px_ref) {
            cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                cv::Scalar(0, 250, 0));
        }
        for(auto &px : goodProjection) {
            cv::rectangle(img2_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                cv::Scalar(0, 250, 0));
        }
        cv::imshow("reference", img1_show);
        cv::imshow("current", img2_show);
        cv::waitKey();
    }

    void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21,
        double fx, double fy, double cx, double cy)
    {

        // parameters
        int pyramids = 4;
        double pyramid_scale = 0.5;
        double scales[] = { 1.0, 0.5, 0.25, 0.125 };

        // create pyramids
        vector<cv::Mat> pyr1, pyr2; // image pyramids
                                    // TODO START YOUR CODE HERE
        cv::Mat pyr_img1 = img1;
        cv::Mat pyr_img2 = img2;
        for(int i = 0; i < pyramids; ++i)
        {
            pyr1.push_back(pyr_img1);
            pyr2.push_back(pyr_img2);
            cv::Size s1(pyr_img1.cols * pyramid_scale, pyr_img1.rows * pyramid_scale);
            cv::Size s2(pyr_img2.cols * pyramid_scale, pyr_img2.rows * pyramid_scale);
            cv::pyrDown(pyr_img1, pyr_img1, s1);
            cv::pyrDown(pyr_img2, pyr_img2, s2);
        }
        // END YOUR CODE HERE

        double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
        for(int level = pyramids - 1; level >= 0; level--) {
            VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
            for(auto &px : px_ref) {
                px_ref_pyr.push_back(scales[level] * px);
            }

            // TODO START YOUR CODE HERE
            // scale fx, fy, cx, cy in different pyramid levels
            fx = fxG * scales[level];
            fy = fyG * scales[level];
            cx = cxG * scales[level];
            cy = cyG * scales[level];
            // END YOUR CODE HERE
            DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
        }
    }
}