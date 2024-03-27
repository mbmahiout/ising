#include "random_mt.h"
#include <cmath>
#include <Eigen/Dense>
#include "utils.h"
#include "models.h"

Eigen::MatrixXd IsingModel::getEffectiveFields() const {
    return m_h + m_J * m_s.cast<double>();
}

Eigen::MatrixXi IsingModel::simulate(int numSims, int numBurn) {
    // burn-in period
    for (int t{0}; t < numBurn; ++t) {
        updateState();
    }

    // simulation
    Eigen::MatrixXi simulation(getNumUnits(), numSims);
    for (int t{0}; t < numSims; ++t) {
        simulation.col(t) = getState();
        updateState();
    }

    return simulation;
}


double EqModel::getEnergyChange(int idx) {
    double sumTerm{0};
    for (int i {0}; i < m_numUnits; ++i) {
        if (i == idx) continue;
        sumTerm += m_J(i, idx) * m_s(i);
    }

    return 2 * m_s(idx) * (sumTerm + m_h(idx));
}

void EqModel::updateState() {
    int idx{Random::get(0, m_numUnits - 1)};
    double energyFlip {getEnergyChange(idx)};
    double probFlip {std::exp(-energyFlip)};
    double u {Misc::getUniformSample(0, 1)};
    if (u < probFlip)
        m_s(idx) *= -1;
}

Eigen::VectorXd NeqModel::getProbActive() const {
    Eigen::MatrixXd effFields {getEffectiveFields()};
    return (1 + (- 2 * effFields.array()).exp()).inverse();
}

void NeqModel::updateState() {
    Eigen::VectorXd probActive {getProbActive()};
    Eigen::VectorXd u {Misc::getUniformVector(m_numUnits, 0, 1)};
    Eigen::VectorXi activeBools {(u.array() < probActive.array()).cast<int>()};
    Eigen::VectorXi newState { 2 * activeBools.array() - 1 };
    setState(newState); // should we move?
}
