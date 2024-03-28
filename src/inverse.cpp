// File: inverse_methods.cpp
#include "inverse.h"

namespace Inverse {
    template <typename T>
    void setMaxLikelihoodParams(T model, Eigen::MatrixXi states) {
        // we also need to take numSims and numBurn for the EqModel... can we do something like **kwargs in Python?
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
