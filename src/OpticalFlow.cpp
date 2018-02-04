#include "OpticalFlow.h"
#include "utils.hpp"

using namespace std;
using namespace cv;

namespace SimpleVO
{
    void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
    ) {

        // parameters
        int half_patch_size = 4;
        int iterations = 10;
        bool have_initial = !kp2.empty();

        for(size_t i = 0; i < kp1.size(); i++) {
            auto kp = kp1[i];
            double dx = 0, dy = 0; // dx,dy need to be estimated
            if(have_initial) {
                dx = kp2[i].pt.x - kp.pt.x;
                dy = kp2[i].pt.y - kp.pt.y;
            }

            double cost = 0, lastCost = 0;
            bool succ = true; // indicate if this point succeeded

            // Gauss-Newton iterations
            for(int iter = 0; iter < iterations; iter++) {
                Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
                Eigen::Vector2d b = Eigen::Vector2d::Zero();
                cost = 0;

                if(kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                    kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {   // go outside
                    succ = false;
                    break;
                }

                // compute cost and jacobian
                for(int x = -half_patch_size; x < half_patch_size; x++)
                    for(int y = -half_patch_size; y < half_patch_size; y++) {
                        double error = 0;
                        Eigen::Vector2d J;  // Jacobian
                        float ptx = kp.pt.x + x;
                        float pty = kp.pt.y + y;
                        if(inverse == false) {
                            // Forward Jacobian
                            error = double(GetPixelValue(img1, ptx, pty) - GetPixelValue(img2, ptx + dx, pty + dy));
                            J[0] = 0.5 * double(GetPixelValue(img2, ptx + dx + 1, pty + dy) - GetPixelValue(img2, ptx + dx - 1, pty + dy));
                            J[1] = 0.5 * double(GetPixelValue(img2, ptx + dx, pty + dy + 1) - GetPixelValue(img2, ptx + dx, pty + dy - 1));
                        }
                        else {
                            // Inverse Jacobian
                            // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                            error = double(GetPixelValue(img1, ptx, pty) - GetPixelValue(img2, ptx + dx, pty + dy));
                            J[0] = 0.5 * double(GetPixelValue(img1, ptx + 1, pty) - GetPixelValue(img1, ptx - 1, pty));
                            J[1] = 0.5 * double(GetPixelValue(img1, ptx, pty + 1) - GetPixelValue(img1, ptx, pty - 1));
                        }

                        // compute H, b and set cost;
                        H += J * J.transpose();
                        b += J * error;
                        cost += error * error;
                    }

                // compute update
                Eigen::Vector2d update;
                update = H.ldlt().solve(b);

                if(isnan(update[0])) {
                    // sometimes occurred when we have a black or white patch and H is irreversible
                    cout << "update is nan" << endl;
                    succ = false;
                    break;
                }
                if(iter > 0 && cost > lastCost) {
                    cout << "cost increased: " << cost << ", " << lastCost << endl;
                    break;
                }

                // update dx, dy
                dx += update[0];
                dy += update[1];
                lastCost = cost;
                succ = true;
            }

            // set kp2
            if(have_initial) {
                kp2[i].pt = kp.pt + Point2f(dx, dy);
                success[i] = succ;
            }
            else {
                KeyPoint tracked = kp;
                tracked.pt += cv::Point2f(dx, dy);
                kp2.push_back(tracked);
                success.push_back(succ);
            }
        }
    }

    void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse) {

        // parameters
        int pyramids = 4;
        double pyramid_scale = 0.5;
        double scales[] = { 1.0, 0.5, 0.25, 0.125 };

        // create pyramids
        vector<Mat> pyr1, pyr2; // image pyramids
        Mat pyr_img1 = img1;
        Mat pyr_img2 = img2;
        for(int i = 0; i < pyramids; i++) {
            pyr1.push_back(pyr_img1);
            pyr2.push_back(pyr_img2);
            Size s1(pyr_img1.cols * pyramid_scale, pyr_img1.rows * pyramid_scale);
            Size s2(pyr_img2.cols * pyramid_scale, pyr_img2.rows * pyramid_scale);
            pyrDown(pyr_img1, pyr_img1, s1);
            pyrDown(pyr_img2, pyr_img2, s2);
        }

        // coarse-to-fine LK tracking in pyramids
        vector<KeyPoint> pyr_kp1;
        for(int i = 0; i < kp1.size(); ++i)
        {
            KeyPoint kp = kp1[i];
            kp.pt *= scales[pyramids - 1];
            pyr_kp1.push_back(kp);
        }

        for(int i = pyramids - 1; i > 0; --i)
        {
            OpticalFlowSingleLevel(pyr1[i], pyr2[i], pyr_kp1, kp2, success, inverse);

            for(int j = 0; j < pyr_kp1.size(); ++j)
            {
                pyr_kp1[j].pt /= pyramid_scale;
                kp2[j].pt /= pyramid_scale;
            }
        }
        OpticalFlowSingleLevel(pyr1[0], pyr2[0], pyr_kp1, kp2, success, inverse);
    }
}