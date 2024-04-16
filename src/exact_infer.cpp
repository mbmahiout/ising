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
    - profile and implement NEQ version
*/

namespace Inverse {

template <typename T>
void updateEMA(
    T& model, 
    Eigen::VectorXd& fieldsEMA, 
    Eigen::MatrixXd& couplingsEMA, 
    const double alpha
) {
    fieldsEMA = alpha * model.getFields() + (1 - alpha) * fieldsEMA;
    couplingsEMA = alpha * model.getCouplings() + (1 - alpha) * couplingsEMA;    
}   

template <typename T>
bool hasConverged(
    T& model,
    Eigen::VectorXd& fieldsEMA, 
    Eigen::MatrixXd& couplingsEMA, 
    const double tolerance
) {
    double fieldsDiff {(model.getFields() - fieldsEMA).norm()};
    double couplingsDiff {(model.getCouplings() - couplingsEMA).norm()};

    bool fieldsConverged {fieldsDiff < tolerance};
    bool couplingsConverged {couplingsDiff < tolerance};

    return (fieldsConverged && couplingsConverged);
}

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





template <typename T>
maxLikelihoodTraj maxLikelihood(
    T& model, 
    Sample& sample, 
    int maxSteps, 
    double learningRate,
    bool useAdam,
    double beta1,
    double beta2,
    double epsilon,
    double alpha,  // for EMA
    double tolerance,  // for convergence checking
    int numSims, 
    int numBurn,
    bool calcLLH
) {
    maxLikelihoodTraj out {};
    
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
            if (numSims > 0) 
                std::tie(dh, dJ) = EqInverse::getGradients(model, sample, numSims, numBurn);
            else
                std::tie(dh, dJ) = EqInverse::getGradients(model, sample);
        }  // else {
        //     std::tie(dh, dJ) = NeqInverse::getGradients(model, sample);
        // }

        Eigen::VectorXd fieldsChange {};
        Eigen::MatrixXd couplingsChange {};

        if (useAdam) {
            Eigen::VectorXd fieldsChange = getAdamParamsChange(dh, h_state, learningRate, beta1, beta2, epsilon);
            Eigen::MatrixXd couplingsChange = getAdamParamsChange(dJ, J_state, learningRate, beta1, beta2, epsilon);
        } else {
            Eigen::VectorXd fieldsChange = learningRate * dh;
            Eigen::MatrixXd couplingsChange = learningRate * dJ;
        }
        
        std::cout << fieldsChange.rows() << 'x' << fieldsChange.cols() << '\n';
        std::cout << model.getFields().rows() << 'x' << model.getFields().cols() << '\n';


        model.setFields(model.getFields() + fieldsChange);
        model.setCouplings(model.getCouplings() + couplingsChange);

        out.fieldsHistory.push_back(model.getFields());
        out.couplingsHistory.push_back(model.getCouplings());

        if (calcLLH) {
            out.LLHs.push_back(model.getLLH(sample));
        }

        updateEMA(model, fieldsEMA, couplingsEMA, alpha);

        if (step > 1) {
            out.fieldsGrads.push_back(dh);
            out.couplingsGrads.push_back(dJ);
            out.fieldsDiffsEMA.push_back(model.getFields() - fieldsEMA);
            out.couplingsDiffsEMA.push_back(model.getCouplings() - couplingsEMA);

            converged = hasConverged(model, fieldsEMA, couplingsEMA, tolerance);
        }

        step += 1;
    }
    
    return out;
}



}

// Explicit template instantiations
template Inverse::maxLikelihoodTraj Inverse::maxLikelihood<EqModel>(
    EqModel& model, 
    Sample& sample, 
    int maxSteps, 
    double learningRate,
    bool useAdam,
    double beta1,
    double beta2,
    double epsilon,
    double alpha,
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
