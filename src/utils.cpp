#include "utils.hpp"
#include <cmath>

using namespace std;
using namespace cv;

namespace SimpleVO
{
    void ComputeDisparity(const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        std::vector<bool>& success,
        std::vector<double>& disparity)
    {
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