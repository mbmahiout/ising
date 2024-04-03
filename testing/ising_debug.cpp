#include "sample.h"
#include "models.h"
#include "inverse.h"
#include "utils.h"
 #include <iostream>
#include <Eigen/Dense>

int main() {
    // true model
    const int numUnits {5};
    const int numSims {20};

    const Eigen::MatrixXd J {Parameters::getGaussianCouplings(numUnits, 0.0, 1.0)};
    const Eigen::VectorXd h {Parameters::getUniformFields(numUnits, -1.3, 1.3)};

    //EqModel true_model {numUnits, J, h};
    NeqModel true_model {numUnits, J, h};

    std::cout << true_model.getFields() << '\n' << true_model.getCouplings() << '\n';

    Sample true_sim {true_model.simulate(numSims)};

    Eigen::VectorXi test {true_sim.getState(0)};

    std::cout << test << '\n';

    // inferred model
    const Eigen::MatrixXd J_init {Parameters::getGaussianCouplings(numUnits, 0.0, 1.0)};
    const Eigen::VectorXd h_init {Parameters::getUniformFields(numUnits, -1.3, 1.3)}; 

    // stop taking numUnits... it's un-necessary
    //EqModel est_model {numUnits, J, h};
    NeqModel est_model {numUnits, J, h};

    /*
        INFERENCE
    */
    int maxSteps {100};
    double lr {0.01};
    int numBurn {1000};

    Inverse::setMaxLikelihoodParams(est_model, true_sim, maxSteps, lr);

    // results
    std::cout << est_model.getFields() << '\n' << est_model.getCouplings() << '\n';


    return 0;
}