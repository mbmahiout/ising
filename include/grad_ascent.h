#ifndef GRAD_ASCENT_H
#define GRAD_ASCENT_H

#include "models.h"
#include "sample.h"
#include <Eigen/Dense>
#include <utility>


namespace Inverse {

struct paramsHistory {
    std::vector<Eigen::VectorXd> fields;
    std::vector<Eigen::MatrixXd> couplings;
};

struct gradsHistory {
    std::vector<Eigen::VectorXd> fieldsGrads;
    std::vector<Eigen::MatrixXd> couplingsGrads;
};

struct statsHistory {
    std::vector<double> avFields;
    std::vector<double> avCouplings;

    std::vector<double> sdFields;
    std::vector<double> sdCouplings;

    std::vector<double> minFields;
    std::vector<double> minCouplings;

    std::vector<double> maxFields;
    std::vector<double> maxCouplings;

    std::vector<double> LLHs;
};

struct gradAscOut {
    paramsHistory params;
    gradsHistory grads;
    statsHistory stats;
};

// the gradient ascent function for maximizing the EQ likelihood or pseudolikelihood, or the NEQ likelihood
template <typename T>
gradAscOut gradientAscent(
    T& model, 
    Sample& sample, 
    int maxSteps,
    double learningRate = 0.1,
    bool useAdam = true,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double epsilon = 1e-5,
    int winSize = 10,
    double tolerance = 1e-5,  // calibrate
    int numSims = 0, 
    int numBurn = 0,
    bool calcLLH=false
);

/*
    AUXILIARY
*/

// functions for dealing with the gradient ascent output




template <typename T>
void updateOutputs(
    const Eigen::VectorXd dh,
    const Eigen::MatrixXd dJ,
    gradAscOut& out,
    const T& model,
    Sample& sample, 
    const int step,
    const int winSize,
    const bool calcLLH

);

template <typename T>
void updateParamStats(
    gradAscOut& out,
    const T& model,
    Sample& sample, 
    const int step,
    const int winSize,
    const bool calcLLH

);


// convergence checking
double getLatestDiff(std::vector<double> v);

double getMaxChange(const gradAscOut& out);

bool hasConverged(const std::vector<double>& maxChanges, const double tolerance, const int winSize);


// adam optimization
template <typename T>
struct AdamState {
    T m;
    T v;
    int t;
};

AdamState<Eigen::VectorXd> initAdamState(const Eigen::VectorXd& params);

AdamState<Eigen::MatrixXd> initAdamState(const Eigen::MatrixXd& params);

template <typename T>
T getAdamParamsChange(
    T& grads,
    AdamState<T>& state,
    double learningRate,
    double beta1,
    double beta2,
    double epsilon
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

#endif //GRAD_ASCENT_H