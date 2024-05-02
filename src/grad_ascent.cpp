#include "grad_ascent.h"
#include "sample.h"
#include "models.h"
#include <Eigen/Dense>
#include <utility>
#include <tuple>
#include <cmath>
#include <type_traits>
#include <algorithm>


namespace Inverse {

template <typename T>
gradAscOut gradientAscent(
    T& model, 
    Sample& sample, 
    int maxSteps, 
    double learningRate,
    bool useAdam,
    double beta1,
    double beta2,
    double epsilon,
    int winSize, 
    double tolerance,
    int numSims, 
    int numBurn,
    bool calcLLH
) {
    gradAscOut out;
    std::vector<double> maxChanges {};
    
    AdamState h_state {initAdamState(model.getFields())};
    AdamState J_state {initAdamState(model.getCouplings())};

    Eigen::VectorXd fieldsEMA {model.getFields()};
    Eigen::MatrixXd couplingsEMA {model.getCouplings()};

    int numUnits {model.getNumUnits()};
    bool converged {false};
    int step {0};
    while (!converged && step <= maxSteps) {

        Eigen::VectorXd dh(numUnits);
        Eigen::MatrixXd dJ(numUnits, numUnits);

        // could be if EqModel, else if NeqModel, else throw exception
        if constexpr (std::is_same_v<T, EqModel>) {
            if (numSims > 0) {
                std::tie(dh, dJ) = EqInverse::getGradients(model, sample, numSims, numBurn);
            } else {
                std::tie(dh, dJ) = EqInverse::getGradients(model, sample);
            }
        } else {
            std::tie(dh, dJ) = NeqInverse::getGradients(model, sample);
        }

        Eigen::VectorXd fieldsChange {};
        Eigen::MatrixXd couplingsChange {};

        if (useAdam) {
            fieldsChange = getAdamParamsChange(dh, h_state, learningRate, beta1, beta2, epsilon);
            couplingsChange = getAdamParamsChange(dJ, J_state, learningRate, beta1, beta2, epsilon);
        } else {
            fieldsChange = learningRate * dh;
            couplingsChange = learningRate * dJ;
        }

        model.setFields(model.getFields() + fieldsChange);
        model.setCouplings(model.getCouplings() + couplingsChange);

        // updating outputs
        updateOutputs(dh, dJ, out, model, sample, step, winSize, calcLLH);

        // convergence checking
        if (step > 2)
            maxChanges.push_back(getMaxChange(out));
        if (step > winSize)
            converged = hasConverged(maxChanges, tolerance, winSize);

        step += 1;
    }

    return out;
}


/*
    AUXILIARY FUNCTIONS
*/

// updating output struct
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

) {
    // updating params history
    out.params.fields.push_back(model.getFields());
    out.params.couplings.push_back(model.getCouplings());

    // updating grads history
    out.grads.fieldsGrads.push_back(dh);
    out.grads.couplingsGrads.push_back(dJ);

    // updating stats history
    updateParamStats(out, model, sample, step, winSize, calcLLH);
}

template <typename T>
void updateParamStats(
    gradAscOut& out,
    const T& model,
    Sample& sample, 
    const int step,
    const int winSize,
    const bool calcLLH

) {
    out.stats.avFields.push_back(model.getFields().mean());
    out.stats.avCouplings.push_back(model.getCouplings().mean());

    out.stats.sdFields.push_back(Misc::getMatrixStd(model.getFields()));    
    out.stats.sdCouplings.push_back(Misc::getMatrixStd(model.getCouplings()));  

    out.stats.minFields.push_back(model.getFields().minCoeff());
    out.stats.minCouplings.push_back(model.getCouplings().minCoeff());

    out.stats.maxFields.push_back(model.getFields().maxCoeff());
    out.stats.maxCouplings.push_back(model.getCouplings().maxCoeff());

    if (calcLLH) {
        out.stats.LLHs.push_back(model.getLLH(sample));
    }
}

// convergence checking
double getLatestDiff(std::vector<double> v) {
    return v.back() - v[v.size() - 2];
}

double getMaxChange(const gradAscOut& out) {
    std::vector<double> absChanges {
        std::abs(getLatestDiff(out.stats.avFields)), 
        std::abs(getLatestDiff(out.stats.avCouplings)), 
        std::abs(getLatestDiff(out.stats.sdFields)), 
        std::abs(getLatestDiff(out.stats.sdCouplings)), 
        std::abs(getLatestDiff(out.stats.minFields)), 
        std::abs(getLatestDiff(out.stats.minCouplings)), 
        std::abs(getLatestDiff(out.stats.maxFields)), 
        std::abs(getLatestDiff(out.stats.maxCouplings))
    };

    return *std::max_element(absChanges.begin(), absChanges.end());
}

bool hasConverged(const std::vector<double>& maxChanges, const double tolerance, const int winSize) {
    std::vector<double> latestChanges {maxChanges.end() - winSize, maxChanges.end()};

    for (auto change : latestChanges) {
        if (change > tolerance) return false;
    }

    return true;
}


// adam optimization
AdamState<Eigen::VectorXd> initAdamState(const Eigen::VectorXd& params) {
    AdamState<Eigen::VectorXd> state;
    state.m = Eigen::VectorXd::Zero(params.size());
    state.v = Eigen::VectorXd::Zero(params.size());
    state.t = 0;
    return state;
}

AdamState<Eigen::MatrixXd> initAdamState(const Eigen::MatrixXd& params) {
    AdamState<Eigen::MatrixXd> state;
    state.m = Eigen::MatrixXd::Zero(params.rows(), params.cols());
    state.v = Eigen::MatrixXd::Zero(params.rows(), params.cols());
    state.t = 0;
    return state;
}

template <typename T>
T getAdamParamsChange(
    T& grads,
    AdamState<T>& state,
    double learningRate,
    double beta1,
    double beta2,
    double epsilon
) {
    state.t++;

    state.m = beta1 * state.m + (1 - beta1) * grads;
    state.v = beta2 * state.v + (1 - beta2) * grads.array().square().matrix();

    T m_hat = state.m / (1 - std::pow(beta1, state.t));
    T v_hat = state.v / (1 - std::pow(beta2, state.t));

    return (learningRate * m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
}

}


// Explicit template instantiations
template Inverse::gradAscOut Inverse::gradientAscent<EqModel>(
    EqModel& model, 
    Sample& sample, 
    int maxSteps, 
    double learningRate,
    bool useAdam,
    double beta1,
    double beta2,
    double epsilon,
    int winSize,
    double tolerance, 
    int numSims, 
    int numBurn,
    bool calcLLH
);

template Inverse::gradAscOut Inverse::gradientAscent<NeqModel>(
    NeqModel& model, 
    Sample& sample, 
    int maxSteps, 
    double learningRate,
    bool useAdam,
    double beta1,
    double beta2,
    double epsilon,
    int winSize,
    double tolerance, 
    int numSims, 
    int numBurn,
    bool calcLLH
);


namespace EqInverse {
std::pair<Eigen::VectorXd, Eigen::MatrixXd> simulationStep(EqModel& model, int numSims, int numBurn) {
    Sample sim {model.simulate(numSims, numBurn)};
    Eigen::VectorXd means {sim.getMeans()};
    Eigen::MatrixXd pcorrs {sim.getPairwiseCorrs()};

    return {means, pcorrs};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> getGradients(EqModel& model, Sample& sample, int numSims, int numBurn) {
    Eigen::MatrixXd pcorrs {sample.getPairwiseCorrs()};
    Eigen::VectorXd means {sample.getMeans()};

    auto [means_sim, pcorrs_sim] = simulationStep(model, numSims, numBurn);
    
    Eigen::MatrixXd dJ {pcorrs - pcorrs_sim};
    Eigen::VectorXd dh {means - means_sim};

    return {dh, dJ};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> getGradients(EqModel& model, Sample& sample) {
    Eigen::MatrixXd states {sample.getStates().cast<double>()};
    Eigen::MatrixXd effFields {model.getEffectiveFields(states)};
    Eigen::MatrixXd tanh_effFields {effFields.unaryExpr([](double elem) { return std::tanh(elem); })};

    Eigen::MatrixXd dh_terms = states - tanh_effFields;  
    Eigen::VectorXd dh = dh_terms.rowwise().mean();

    int numBins {sample.getNumBins()};
    int numUnits {model.getNumUnits()};
    Eigen::MatrixXd dJ = Eigen::MatrixXd::Zero(numUnits, numUnits);

    for (int t = 0; t < numBins; ++t) {
        dJ.noalias() += dh_terms.col(t) * states.col(t).transpose();
    }
    dJ /= numBins;
    dJ.diagonal().setZero();

    return {dh, dJ};
}

}

namespace NeqInverse {

// review logic and debug
std::pair<Eigen::VectorXd, Eigen::MatrixXd> getGradients(NeqModel& model, Sample& sample) {
    int numBins {sample.getNumBins()};
    int numUnits {model.getNumUnits()};

    Eigen::MatrixXd states {sample.getStates().cast<double>()};
    Eigen::MatrixXd statesForwardShifted {states(Eigen::all, Eigen::seq(1, Eigen::last))};
    Eigen::MatrixXd statesBackwardShifted {states(Eigen::all, Eigen::seq(0, Eigen::last-1))};

    Eigen::MatrixXd effFields {model.getEffectiveFields(statesBackwardShifted)};
    Eigen::MatrixXd tanh_effFields {effFields.unaryExpr([](double elem) { return std::tanh(elem); })};

    Eigen::MatrixXd dh_terms = statesForwardShifted - tanh_effFields;  
    Eigen::VectorXd dh = dh_terms.rowwise().mean();

    Eigen::MatrixXd dJ = Eigen::MatrixXd::Zero(numUnits, numUnits);
    for (int t = 0; t < numBins; ++t) {
        dJ.noalias() += dh_terms.col(t+1) * states.col(t).transpose();
    }
    dJ /= numBins;
    dJ.diagonal().setZero();

    return {dh, dJ};
}

}
