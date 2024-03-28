#ifndef OBSERVABLES_H
#define OBSERVABLES_H
#include <Eigen/Dense>

Eigen::VectorXd getMeans(const Eigen::MatrixXi& states);

Eigen::MatrixXd getPairwiseCorrs(const Eigen::MatrixXi& states);

Eigen::MatrixXd getConnectedCorrs(const Eigen::MatrixXi& states);

Eigen::MatrixXd getDelayedCorrs(const Eigen::MatrixXi& states, int dt);

#endif //OBSERVABLES_H
