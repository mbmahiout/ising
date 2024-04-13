#include "exact_infer.h"
#include "sample.h"
#include "models.h"
#include <Eigen/Dense>
#include <utility>
#include <tuple>
#include <cmath>
#include <type_traits>

/*
Next steps:
    - include convergence criterion for maxLikelihood
    - implement getLLH (maybe only compute, say, every 10 steps)
    - profile and implement NEQ version
*/

namespace Inverse {

template <typename T>
maxLikelihoodTraj maxLikelihood(
    T& model, 
    Sample& sample, 
    int maxSteps, 
    double learningRate, 
    int numSims, 
    int numBurn,
    bool calcLLH
) {
    maxLikelihoodTraj out {};

    int numUnits {model.getNumUnits()};
    bool converged {false};
    int step {0};
    while (!converged && step <= maxSteps) {
        step += 1;
        Eigen::VectorXd dh(numUnits);
        Eigen::MatrixXd dJ(numUnits, numUnits);

        if constexpr (std::is_same_v<T, EqModel>) {
            if (numSims > 0) 
                std::tie(dh, dJ) = EqInverse::getGradients(model, sample, numSims, numBurn);
            else
                std::tie(dh, dJ) = EqInverse::getGradients(model, sample);
        // } else {
        //     std::tie(dh, dJ) = NeqInverse::getGradients(model, sample);
        // }
        
        model.setFields(model.getFields() + learningRate * dh);
        model.setCouplings(model.getCouplings() + learningRate * dJ);

        out.fieldsHistory.push_back(model.getFields());
        out.couplingsHistory.push_back(model.getCouplings());
        // if (calcLLH) { // to-do (maybe calculate eg, every 10 steps only)
        //     out.LLHs.push_back(getLLH(model, sample))
        // }
    }
}

    return out;
}

}

// Explicit template instantiations
template Inverse::maxLikelihoodTraj Inverse::maxLikelihood<EqModel>(
    EqModel& model, Sample& sample, int maxSteps, double learningRate, int numSims, int numBurn, bool calcLLH
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
    Eigen::MatrixXd states_transposed {states.transpose()};
   
    Eigen::MatrixXd effFields {model.getEffectiveFields(states)};
    Eigen::MatrixXd tanh_effFields {effFields.array().tanh().matrix()};

    Eigen::MatrixXd dh_terms {states - tanh_effFields};  
    int numBins {sample.getNumBins()}; 
    Eigen::VectorXd dh {dh_terms.rowwise().sum() / numBins};

    int numUnits {model.getNumUnits()};
    Eigen::MatrixXd dJ {Eigen::MatrixXd::Zero(numUnits, numUnits)};
    for (int t {0}; t < numBins; ++t) {
        dJ += dh_terms.col(t) * states.col(t).transpose();
    }
    dJ /= numBins;
    return {dh, dJ};
}

}

// namespace NeqInverse {

// std::pair<Eigen::VectorXd, Eigen::MatrixXd> getGradients(NeqModel& model, Sample& sample) {
//     int numUnits {model.getNumUnits()};
//     int numBins {sample.getNumBins()};

//     Eigen::VectorXd dh {Eigen::VectorXd::Zero(numUnits)};
//     Eigen::MatrixXd dJ {Eigen::MatrixXd::Zero(numUnits, numUnits)};

//     for (int t {0}; t < numBins - 1; ++t) {
//         Eigen::VectorXi state_t {sample.getState(t)};
//         Eigen::VectorXi state_tp1 {sample.getState(t+1)};

//         Eigen::VectorXd effFields {model.getEffectiveFields(state_t)};
//         Eigen::VectorXd dh_term {state_tp1.cast<double>() - effFields.array().tanh().matrix()};

//         dh += dh_term;
//         dJ += dh_term * (state_t.cast<double>()).transpose();
//     }
//     dh /= numBins;
//     dJ /= numBins;

//     return {dh, dJ};
// }

// }
