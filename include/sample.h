#ifndef SAMPLE_H
#define SAMPLE_H

#include <Eigen/Dense>

class Sample
{
private:
    Eigen::MatrixXi m_states{};
    int m_numBins{};
    int m_numUnits{};
    mutable Eigen::VectorXd m_means{};
    mutable Eigen::MatrixXd m_pcorrs{};
    mutable Eigen::MatrixXd m_ccorrs{};
    mutable Eigen::MatrixXd m_dcorrs{};
    // mutable Eigen::MatrixXd m_tricorrs{};

    // flags
    mutable bool m_isMeansSet{false};
    mutable bool m_isPcorrsSet{false};
    mutable bool m_isCcorrsSet{false};

    // setters
    void setMeans() const;
    void setPairwiseCorrs() const;
    void setConnectedCorrs() const;
    void setDelayedCorrs(int dt) const;
    // void setTriCorrs() const;

public:
    Sample(Eigen::MatrixXi states)
        : m_states{states}, m_numBins{static_cast<int>(states.cols())}, m_numUnits{static_cast<int>(states.rows())}, m_isMeansSet{false}, m_isPcorrsSet{false}, m_isCcorrsSet{false}, m_isDcorrsSet{false}
    {
    }

    // getters
    [[nodiscard]] int getNumUnits() const { return m_numUnits; }
    [[nodiscard]] int getNumBins() const { return m_numBins; }
    [[nodiscard]] Eigen::MatrixXi getStates() const { return m_states; }
    [[nodiscard]] Eigen::VectorXi getState(int t) const { return m_states.col(t); }

    [[nodiscard]] Eigen::VectorXd getMeans() const
    {
        if (!m_isMeansSet)
        {
            setMeans();
        }

        return m_means;
    }

    [[nodiscard]] Eigen::MatrixXd getPairwiseCorrs() const
    {
        if (!m_isPcorrsSet)
        {
            setPairwiseCorrs();
        }
        return m_pcorrs;
    }

    [[nodiscard]] Eigen::MatrixXd getConnectedCorrs() const
    {
        if (!m_isCcorrsSet)
        {
            setConnectedCorrs();
        }
        return m_ccorrs;
    }

    [[nodiscard]] Eigen::MatrixXd getDelayedCorrs(int dt) const
    {
        setDelayedCorrs(dt);
        return m_dcorrs;
    }

    // [[nodiscard]] Eigen::MatrixXd getTriCorrs() const
    // {
    //     if (!m_isTriCorrsSet)
    //     {
    //         setTriCorrs();
    //     }
    //     return m_tricorrs;
    // }
};

#endif // SAMPLE_H