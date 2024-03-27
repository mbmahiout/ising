#ifndef ISING_OBSERVABLES_H
#define ISING_OBSERVABLES_H
#include <Eigen/Dense>

Eigen::MatrixXd getPairwiseCorrs(const Eigen::MatrixXi& states);

#endif //ISING_OBSERVABLES_H
