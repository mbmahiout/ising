#ifndef UTILS_H
#define UTILS_H

#include "random_mt.h"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "utils.h"

namespace Misc
{
    template <typename T>
    double getMatrixStd(const T &m)
    {
        double mean{m.mean()};
        double sumOfSquares{(m.array() - mean).square().sum()};
        return std::sqrt(sumOfSquares / (m.size() - 1));
    }

    double getUniformSample(const double leftEdge, const double rightEdge);

    Eigen::VectorXd getUniformVector(const int len, const double leftEdge, const double rightEdge);

    Eigen::VectorXi getRandomState(int numUnits);
}

namespace Parameters
{
    Eigen::MatrixXd getGaussianCouplings(int numUnits, double mu, double sigma);

    Eigen::VectorXd getUniformFields(int numUnits, double leftEdge, double rightEdge);
}

#endif // UTILS_H
