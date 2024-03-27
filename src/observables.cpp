#include "observables.h"
#include <Eigen/Dense>

Eigen::MatrixXd getPairwiseCorrs(const Eigen::MatrixXi& states) {   // pass by const ref or move!
    int numBins {static_cast<int>(states.cols())};
    Eigen::MatrixXd C {(states * states.transpose()).cast<double>() / numBins};
    return C;
}

// import into python and compare (make a new file for this)