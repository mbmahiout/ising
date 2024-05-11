#include "sample.h"
#include <Eigen/Dense>

void Sample::setMeans() const
{
    m_means = (m_states.cast<double>()).rowwise().mean();
    m_isMeansSet = true;
}

void Sample::setPairwiseCorrs() const
{
    m_pcorrs = (m_states * m_states.transpose()).cast<double>() / (m_numBins - 1);
    m_isPcorrsSet = true;
}

void Sample::setConnectedCorrs() const
{
    Eigen::VectorXd m{getMeans()};
    Eigen::MatrixXd chi{getPairwiseCorrs()};
    m_ccorrs = chi - m * m.transpose();
    m_isCcorrsSet = true;
}

void Sample::setDelayedCorrs(int dt) const
{
    int numBinsNew{m_numBins - dt};
    Eigen::MatrixXi states_head{m_states.leftCols(numBinsNew)};
    Eigen::MatrixXi states_tail{m_states.rightCols(numBinsNew)};
    Eigen::VectorXd m_head{(states_head.cast<double>()).rowwise().mean()};
    Eigen::VectorXd m_tail{(states_tail.cast<double>()).rowwise().mean()};
    Eigen::MatrixXd D{(states_head * states_tail.transpose()).cast<double>() / numBinsNew};
    m_dcorrs = D - m_head * m_tail.transpose();
    m_isDcorrsSet = true;
}