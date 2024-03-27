#ifndef ISING_MODELS_H
#define ISING_MODELS_H

#include "random_mt.h"
#include <cmath>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "utils.h"

class IsingModel {
protected:
    const int m_numUnits {};
    const Eigen::MatrixXd m_J {};
    const Eigen::VectorXd m_h {};
    Eigen::VectorXi m_s {};

    virtual void updateState() = 0;


public:
    IsingModel(const int numUnits, Eigen::MatrixXd J, Eigen::VectorXd h)
            : m_numUnits{numUnits}
            , m_J{std::move(J)}
            , m_h{std::move(h)}
            , m_s{Misc::getRandomState(numUnits)}
            {}

    virtual ~IsingModel() = default;

    [[nodiscard]] int getNumUnits() const { return m_numUnits; }
    [[nodiscard]] Eigen::VectorXd getFields() const { return m_h; }
    [[nodiscard]] Eigen::MatrixXd getCouplings() const { return m_J; }
    [[nodiscard]] Eigen::VectorXi getState() const { return m_s; }

    void setState(const Eigen::VectorXi &s) { m_s = s; }

    [[nodiscard]] Eigen::MatrixXd getEffectiveFields() const;

    Eigen::MatrixXi simulate(int numSims, int numBurn=1000);
};

class EqModel: public IsingModel {
public:
    EqModel(const int numUnits, Eigen::MatrixXd J, Eigen::VectorXd h)
    : IsingModel(numUnits, std::move(J), std::move(h))
    {}

    double getEnergyChange(int idx);
    void updateState() override;
};

class NeqModel: public IsingModel {
public:
    NeqModel(const int numUnits, Eigen::MatrixXd J, Eigen::VectorXd h)
            : IsingModel(numUnits, std::move(J), std::move(h))
    {}

    [[nodiscard]] Eigen::VectorXd getProbActive() const;
    void updateState() override;
};


#endif //ISING_MODELS_H
