#include "random_mt.h"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "utils.h"


namespace Misc {
    double getUniformSample(const double leftEdge, const double rightEdge) {
        std::uniform_real_distribution<double> uniformDist(leftEdge, rightEdge);
        return uniformDist(Random::mt);
    }

    Eigen::VectorXd getUniformVector(const int len, const double leftEdge, const double rightEdge) {
        Eigen::VectorXd u(len);
        for (int i {0}; i < len; ++i) {
            u(i) = Misc::getUniformSample(leftEdge, rightEdge);
        }

        return u;
    }

    Eigen::VectorXi getRandomState(int numUnits) {
        Eigen::VectorXi s(numUnits);
        for (int i {0}; i < numUnits; ++i) {
            int u {Random::get(0,1)};
            s(i) = 2 * u - 1;
        }

        return s;
    }
}


namespace Parameters {
    Eigen::MatrixXd getGaussianCouplings(int numUnits, double mu, double sigma) { // pass consts?
        Eigen::MatrixXd J(numUnits, numUnits);
        J.setZero();
        std::normal_distribution<double> normalDist(mu, sigma);
        for (int i {0}; i < numUnits; ++i) {
            for (int j {i+1}; j < numUnits; ++j) {
                J(i,j) = normalDist(Random::mt);
                J(j,i) = J(i,j);
            }
        }

        return J;
    }

    Eigen::VectorXd getUniformFields(int numUnits, double leftEdge, double rightEdge) { // pass consts?
        Eigen::VectorXd h(numUnits);
        for (int i {0}; i < numUnits; ++i) {
            h(i) = Misc::getUniformSample(leftEdge, rightEdge);
        }

        return h;
    }
}