#include "utils.hpp"
#include <cmath>

using namespace std;
using namespace cv;

namespace SimpleVO
{
    void ConvertKeyPointToEigen(const std::vector<cv::KeyPoint>& kp, VecVector2d& vec2d)
    {
        if(vec2d.size() > 0)
        {
            vec2d.clear();
        }

        for(unsigned int i = 0; i < kp.size(); ++i)
        {
            Eigen::Vector2d vec(kp[i].pt.x, kp[i].pt.y);
            vec2d.push_back(vec);
        }
    }

    void ConvertEigenToKeyPoint(const VecVector2d& vec2d, std::vector<cv::KeyPoint>& kp)
    {
        if(kp.size() > 0)
        {
            kp.clear();
        }

        for(unsigned int i = 0; i < vec2d.size(); ++i)
        {
            cv::KeyPoint p(vec2d[i][0], vec2d[i][1], 1.0);
            kp.push_back(p);
        }
    }

    void ComputeDisparity(const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        std::vector<bool>& success,
        std::vector<double>& disparity)
    {
        if(!disparity.empty())
        {
            disparity.clear();
        }

        // delete points not satisfy constraint
        for(unsigned int i = 0; i < success.size(); ++i)
        {
            // offset on y direction too large
            double yTh = 1.0;
            if(abs(kp1[i].pt.y - kp2[i].pt.y) >= yTh)
            {
                success[i] = false;
            }

            // points on left image should have larger x than points on right image
            double disp = double(kp1[i].pt.x - kp2[i].pt.x);
            if(disp < 0)
            {
                success[i] = false;
                disp = 0.0;
            }

            disparity.push_back(disp);
        }
    }
}