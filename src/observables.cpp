#include "observables.h"
#include <Eigen/Dense>

Eigen::VectorXd getMeans(const Eigen::MatrixXi& states) {
    return (states.cast<double>()).rowwise().mean();
}

Eigen::MatrixXd getPairwiseCorrs(const Eigen::MatrixXi& states) { 
    int numBins {static_cast<int>(states.cols())};

    return (states * states.transpose()).cast<double>() / numBins;
}

/*
    TO-DO:
    1. include defs in pybinindgs file
    2. test that all the methods give the right answers (compare with corresponding python methods)
    3. get started on the inverse methods!
*/

Eigen::MatrixXd getConnectedCorrs(const Eigen::MatrixXi& states) { 
    Eigen::VectorXd m {getMeans(states)};
    Eigen::MatrixXd C {getPairwiseCorrs(states)};
    
    return C - m * m.transpose();
}

Eigen::MatrixXd getDelayedCorrs(const Eigen::MatrixXi& states, int dt) {
    int numBins {static_cast<int>(states.cols())};
    int numBinsNew {numBins - dt};
    
    Eigen::MatrixXi states_head {states.rightCols(numBinsNew)};
    Eigen::MatrixXi states_tail {states.leftCols(numBinsNew)};
    
    Eigen::VectorXd m_head {getMeans(states_head)};
    Eigen::VectorXd m_tail {getMeans(states_tail)};

    Eigen::MatrixXd D {states_head * states_tail.transpose() / numBinsNew};
    return D - m_head * m_tail.transpose();
}