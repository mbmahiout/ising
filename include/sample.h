#ifndef SAMPLE_H
#define SAMPLE_H

#include <Eigen/Dense>

class Sample {
private:
    Eigen::MatrixXi m_states {};
    int m_numBins {};
    int m_numUnits {};
    mutable Eigen::VectorXd m_means {};
    mutable Eigen::MatrixXd m_pcorrs {};
    mutable Eigen::MatrixXd m_ccorrs {};
    mutable Eigen::MatrixXd m_dcorrs {};

    //flags
    mutable bool m_isMeansSet {false};
    mutable bool m_isPcorrsSet {false};
    mutable bool m_isCcorrsSet {false};
    mutable bool m_isDcorrsSet {false};

    // setters
    void setMeans() const;
    void setPairwiseCorrs() const;
    void setConnectedCorrs() const;
    void setDelayedCorrs(int dt) const;

public:
    Sample(Eigen::MatrixXi states) 
    : m_states {states}
    , m_numBins {static_cast<int>(states.cols())}
    , m_numUnits {static_cast<int>(states.rows())}
    , m_isMeansSet {false}
    , m_isPcorrsSet {false}
    , m_isCcorrsSet {false}
    , m_isDcorrsSet {false}
    {}

    
    // getters
    [[nodiscard]] Eigen::MatrixXi getStates() const {return m_states;}

    [[nodiscard]] Eigen::VectorXd getMeans() const {
        if (!m_isMeansSet) {
            setMeans();
        }
        
        return m_means;
    }

    [[nodiscard]] Eigen::MatrixXd getPairwiseCorrs() const {
        if (!m_isPcorrsSet) {
            setPairwiseCorrs();
        }
        return m_pcorrs;
    }

    [[nodiscard]] Eigen::MatrixXd getConnectedCorrs() const {
        if (!m_isCcorrsSet) {
            setConnectedCorrs();
        }
        return m_ccorrs;
    }

    [[nodiscard]] Eigen::MatrixXd getDelayedCorrs(int dt) const {
        if (!m_isDcorrsSet) {
            setDelayedCorrs(dt);
        }
        return m_dcorrs;
    }
};

#endif //SAMPLE_H