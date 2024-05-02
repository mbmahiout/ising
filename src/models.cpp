#include "random_mt.h"
#include "sample.h"
#include <cmath>
#include <Eigen/Dense>
#include "utils.h"
#include "models.h"
#include <boost/dynamic_bitset.hpp>

// abstract base class
Eigen::VectorXd IsingModel::getEffectiveFields(Eigen::VectorXd stateDouble) const {
    return m_h + m_J * stateDouble;
}

Eigen::MatrixXd IsingModel::getEffectiveFields(Eigen::MatrixXd statesDouble) const {
    Eigen::MatrixXd effFields {m_J * statesDouble};
    effFields.colwise() += m_h;
    return effFields;
}

 Sample IsingModel::simulate(int numSims, int numBurn) {
    // burn-in period
    for (int t{0}; t < numBurn; ++t) {
        updateState();
    }

    // simulation
    Eigen::MatrixXi simStates(getNumUnits(), numSims);
    for (int t{0}; t < numSims; ++t) {
        simStates.col(t) = getState();
        updateState();
    }

    Sample simulation(simStates);
    return simulation;
}

// equilibrium model
double EqModel::getEnergyChange(int idx) const {
    double sumTerm{0};
    for (int i {0}; i < m_numUnits; ++i) {
        if (i == idx) continue;
        sumTerm += m_J(i, idx) * m_state(i);
    }

    return 2 * m_state(idx) * (sumTerm + m_h(idx));
}

void EqModel::updateState() {
    int idx{Random::get(0, m_numUnits - 1)};
    double energyFlip {getEnergyChange(idx)};
    double probFlip {std::exp(-energyFlip)};
    double u {Misc::getUniformSample(0, 1)};
    if (u < probFlip)
        m_state(idx) *= -1;
}

Eigen::MatrixXi EqModel::genFullStateSpace() const {
    Eigen::Index numStates {1LL << m_numUnits}; // = 2^N; long long to avoid overflow
    Eigen::MatrixXi states(m_numUnits, numStates);

    for (Eigen::Index s {0}; s < numStates; ++s) {
        boost::dynamic_bitset<> bits(m_numUnits, s); // dynamic bitset of size numUnits initialized to s
        for (int i {0}; i < m_numUnits; ++i) {
            states(i, s) = bits[i] ? 1 : -1;
        }
    }

    return states;
}

double EqModel::getHamiltonian(const Eigen::VectorXi& state) const {
    double H {-m_h.dot(state.cast<double>())};  // inner product?
    
    for (int i {0}; i < m_numUnits; ++i) {
        for (int j {i + 1}; j < m_numUnits; ++j) {
            H -= m_J(i, j) * state(i) * state(j);
        }
    }

    return H;
}

double EqModel::getPartitionFunc() const {
    double Z {0};
    
    Eigen::MatrixXi stateSpace {genFullStateSpace()};
    for (int s {0}; s < stateSpace.cols(); ++s) {
        Eigen::VectorXi state {stateSpace.col(s)}; 
        double H {getHamiltonian(state)};
        Z += std::exp(-H);
    }

    return Z;
}

double EqModel::getLLH(Sample& sample) const {
    Eigen::VectorXd m {sample.getMeans()};
    Eigen::MatrixXd chi {sample.getPairwiseCorrs()};
    double Z {getPartitionFunc()};

    double LLH {m.sum() - std::log(Z)};
    for (int i {0}; i < m_numUnits; ++i) {
        for (int j {i + 1}; j < m_numUnits; ++j) {
            LLH += m_J(i, j) * chi(i, j);
        }
    }

    return LLH;    
}

// non-equilibrium model
Eigen::VectorXd NeqModel::getProbActive() const {
    Eigen::VectorXd stateDouble {m_state.cast<double>()};
    Eigen::VectorXd effFields {getEffectiveFields(stateDouble)};
    return (1 + (- 2 * effFields.array()).exp()).inverse();
}

void NeqModel::updateState() {
    Eigen::VectorXd probActive {getProbActive()};
    Eigen::VectorXd u {Misc::getUniformVector(m_numUnits, 0, 1)};
    Eigen::VectorXi activeBools {(u.array() < probActive.array()).cast<int>()};
    Eigen::VectorXi newState { 2 * activeBools.array() - 1 };
    setState(newState); // should we move?
}

