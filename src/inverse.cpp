// File: inverse_methods.cpp
#include "inverse.h"

namespace Inverse {

template <typename T>
void setMaxLikelihoodParams(
    T model, Eigen::MatrixXi states, int maxSteps, int numSims, int numBurn
) {
    bool converged {false};
    int step {0};
    while (!converged && step <= max_steps) {
        step += 1;
        if (numSims > 0) 
            updateParameters(model, states, numSims, numBurn)
        else
            updateParameters(model, states)
    }     
}

}

namespace EqInverse {

void updateParameters(EqModel& model, const Eigen::MatrixXi& states, int numSims, int numBurn) {
    
}

void updateParameters(EqModel& model, const Eigen::MatrixXi& states) {

}

}

namespace NeqInverse {

void updateParameters(NeqModel& model, const Eigen::MatrixXi& states) {

}

}
