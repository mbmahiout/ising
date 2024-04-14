#ifndef EXACT_INFER_H
#define EXACT_INFER_H

#include "models.h"
#include "sample.h"
#include <Eigen/Dense>
#include <utility>


namespace Inverse {

struct maxLikelihoodTraj {  // sizes could be preallocated
    // parameters trajectory
    std::vector<Eigen::VectorXd> fieldsHistory;
    std::vector<Eigen::MatrixXd> couplingsHistory;

    // for convergence testing
    std::vector<Eigen::VectorXd> fieldsDiffsEMA;
    std::vector<Eigen::MatrixXd> couplingsDiffsEMA;

    std::vector<Eigen::VectorXd> fieldsGrads;
    std::vector<Eigen::MatrixXd> couplingsGrads;

    // optionally (for testing)
    std::vector<double> LLHs;
};

template <typename T>
void updateEMA(
    T& model, 
    Eigen::VectorXd& fieldsEMA, 
    Eigen::MatrixXd& couplingsEMA, 
    const double alpha
);

template <typename T>
bool hasConverged(
    T& model,
    Eigen::VectorXd& fieldsEMA, 
    Eigen::MatrixXd& couplingsEMA, 
    const double tolerance
);

template <typename T>
maxLikelihoodTraj maxLikelihood(
    T& model, 
    Sample& sample, 
    int maxSteps,
    double learningRate=0.1,  // calibrate 
    double alpha=0.1,  // calibrate
    double tolerance=1e-5,  // calibrate
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