#ifndef ISING_OBSERVABLES_H
#define ISING_OBSERVABLES_H
#include <Eigen/Dense>

Eigen::VectorXd getMeans(const Eigen::MatrixXi& states);

Eigen::MatrixXd getPairwiseCorrs(const Eigen::MatrixXi& states);

#endif //ISING_OBSERVABLES_H
