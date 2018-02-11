#include "optimization.h"

#include <iostream>
#include "local_parameterization_se3.hpp"

using namespace std;

namespace Eigen {
    namespace internal {

        template <class T, int N, typename NewType>
        struct cast_impl<ceres::Jet<T, N>, NewType> {
            EIGEN_DEVICE_FUNC
                static inline NewType run(ceres::Jet<T, N> const& x) {
                return static_cast<NewType>(x.a);
            }
        };
    }
}

namespace SimpleVO
{
    struct ProjectCostFunctor {
        ProjectCostFunctor(double _fx, double _fy, double _cx, double _cy,
            double _px, double _py) : fx(_fx), fy(_fy), cx(_cx), cy(_cy),
        px(_px), py(_py) {}

        template <class T>
        bool operator()(T const* const sP3d, T const* const sPose, T* sResiduals) const {
            Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p3d(sP3d);
            Eigen::Map<Sophus::SE3<T> const> const pose(sPose);
            Eigen::Map<Eigen::Matrix<T, 2, 1> > residuals(sResiduals);
            Eigen::Matrix<double, 2, 1> measure(px, py);

            Eigen::Matrix<T, 3, 1> p3d_trans = pose * p3d;
            p3d_trans = p3d_trans / p3d_trans(2);
            Eigen::Matrix<T, 2, 1> proj(fx * p3d_trans(0) + cx, 
                fy * p3d_trans(1) + cy);

            residuals = measure.cast<T>() - proj;
            return true;
        }

        // internal parameters
        double fx, fy, cx, cy;

        // observation
        double px, py;
    };

    void Optimize::SetIntrinsic(double _fx, double _fy, double _cx, double _cy)
    {
        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;
    }

    void Optimize::AddParameters(Sophus::SE3d& pose)
    {
        // Specify local update rule for our parameter
        problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters,
            new Sophus::LocalParameterizationSE3);
    }

    void Optimize::AddOvservation(double px, double py,
        Eigen::Matrix<double, 3, 1>& p3d,
        Sophus::SE3d& pose)
    {
        // Create and add cost function. Derivatives will be evaluated via
        // automatic differentiation
        ProjectCostFunctor* c = new ProjectCostFunctor(fx, fy, cx, cy, px, py);
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ProjectCostFunctor, 2, 3,
            Sophus::SE3d::num_parameters>(c);
        problem.AddResidualBlock(cost_function, NULL, p3d.data(), pose.data());
    }

    void Optimize::SetOptions()
    {
        options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
        options.max_num_iterations = 100;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    }

    void Optimize::Run()
    {
        Solve(options, &problem, &summary);
        cout << summary.BriefReport() << endl;
    }
}