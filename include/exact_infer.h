#ifndef EXACT_INFER_H
#define EXACT_INFER_H

#include "models.h"
#include "sample.h"
#include <Eigen/Dense>
#include <utility>


namespace Inverse {

struct maxLikelihoodTraj {
    std::vector<Eigen::VectorXd> fieldsHistory;
    std::vector<Eigen::MatrixXd> couplingsHistory;
    std::vector<double> LLHs; 
};

template <typename T>
maxLikelihoodTraj maxLikelihood(
    T& model, 
    Sample& sample, 
    int maxSteps, 
    double learningRate, 
    int numSims=0, 
    int numBurn=0,
    bool calcLLH=false
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

// namespace NeqInverse {

// std::pair<Eigen::VectorXd, Eigen::MatrixXd> getGradients(NeqModel& model, Sample& sample);

// }

#endif //EXACT_INFER_H