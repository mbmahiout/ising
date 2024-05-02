#ifndef MODELS_H
#define MODELS_H

#include "random_mt.h"
#include "sample.h"
#include <cmath>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "utils.h"

class IsingModel {
protected:
    Eigen::MatrixXd m_J {};
    Eigen::VectorXd m_h {};
    const int m_numUnits {};
    Eigen::VectorXi m_state {};

    virtual void updateState() = 0;

public:
    IsingModel(const Eigen::MatrixXd& J, const Eigen::VectorXd& h)
            : m_J {J}
            , m_h {h}
            , m_numUnits {static_cast<int>(h.rows())}
            , m_state {Misc::getRandomState(m_numUnits)}
            {}

    virtual ~IsingModel() = default;

    // setters (compare with passing by const reference)
    void setState(const Eigen::VectorXi &state) {m_state = state;}
    void setFields(const Eigen::VectorXd& h) {m_h = h;}
    void setCouplings(const Eigen::MatrixXd& J) {m_J = J;}

    // getters
    [[nodiscard]] int getNumUnits() const {return m_numUnits;}
    [[nodiscard]] Eigen::VectorXd getFields() const {return m_h;}
    [[nodiscard]] Eigen::MatrixXd getCouplings() const {return m_J;}
    [[nodiscard]] Eigen::VectorXi getState() const {return m_state;}
    [[nodiscard]] Eigen::VectorXd getEffectiveFields(Eigen::VectorXd stateDouble) const;
    [[nodiscard]] Eigen::MatrixXd getEffectiveFields(Eigen::MatrixXd statesDouble) const;

    // MCMC simulation
    Sample simulate(int numSims, int numBurn=1000);
};

class EqModel: public IsingModel {
public:
    EqModel(const Eigen::MatrixXd& J, const Eigen::VectorXd& h)
    : IsingModel(J, h)
    {}

    // functions for simulation
    double getEnergyChange(int idx) const; 
    void updateState() override;

    // functions for likelihood calculation
    Eigen::MatrixXi genFullStateSpace() const; 
    double getHamiltonian(const Eigen::VectorXi& state) const;
    double getPartitionFunc() const;
    double getLLH(Sample& sample) const;

};

class NeqModel: public IsingModel {
public:
    NeqModel(const Eigen::MatrixXd& J, const Eigen::VectorXd& h)
    : IsingModel(J, h)
    {}

    // functions for simulation
    Eigen::VectorXd getProbActive() const;
    void updateState() override;

    // functions for likelihood calculation
    double getLLH(Sample& sample) const;

};

#endif //MODELS_H