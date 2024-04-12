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
    const int m_numUnits {};
    Eigen::MatrixXd m_J {};
    Eigen::VectorXd m_h {};
    Eigen::VectorXi m_state {};

    virtual void updateState() = 0;

public:
    IsingModel(const int numUnits, const Eigen::MatrixXd& J, const Eigen::VectorXd& h)
            : m_numUnits{numUnits} // doesn't have to be explicit; can be inferred from h
            , m_J {J}
            , m_h {h}
            , m_state {Misc::getRandomState(numUnits)} // ^though maybe could cause issue here?
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
    EqModel(const int numUnits, const Eigen::MatrixXd& J, const Eigen::VectorXd& h)
    : IsingModel(numUnits, J, h)
    {}

    double getEnergyChange(int idx);
    void updateState() override;
};

// class NeqModel: public IsingModel {
// public:
//     NeqModel(const int numUnits, const Eigen::MatrixXd& J, const Eigen::VectorXd& h)
//             : IsingModel(numUnits, J, h)
//     {}

//     [[nodiscard]] Eigen::VectorXd getProbActive() const;
//     void updateState() override;
// };


#endif //MODELS_H