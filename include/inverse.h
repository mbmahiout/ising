#ifndef INVERSE_H
#define INVERSE_H

#include "models.h"
#include "sample.h"
#include <Eigen/Dense>
#include <utility>


namespace Inverse {

template <typename T>
void setMaxLikelihoodParams(
    T& model, Sample& sample, int maxSteps, double learningRate, int numSims=0, int numBurn=0
);

}

namespace EqInverse {

std::pair<Eigen::VectorXd, Eigen::MatrixXd> simulationStep(
    EqModel& model, int numSims, int numBurn
);

std::pair<Eigen::VectorXd, Eigen::MatrixXd> getGradients(
    EqModel& model, Sample& sample, int numSims, int numBurn
);

std::pair<Eigen::VectorXd, Eigen::MatrixXd> getGradients(EqModel& model, Sample& sample);

}

namespace NeqInverse {

std::pair<Eigen::VectorXd, Eigen::MatrixXd> getGradients(NeqModel& model, Sample& sample);

}

#endif //INVERSE_H