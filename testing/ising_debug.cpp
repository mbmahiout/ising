#include "sample.h"
#include "models.h"
#include "exact_infer.h"
#include "utils.h"
#include "timer.h"
 #include <iostream>
#include <Eigen/Dense>



int main() {
    // true model
    const int numUnits {3};
    const int numSims {15000};

    const Eigen::MatrixXd J {Parameters::getGaussianCouplings(numUnits, 0.0, 1.0)};
    const Eigen::VectorXd h {Parameters::getUniformFields(numUnits, -1.3, 1.3)};

    EqModel true_model {J, h};

    std::cout << true_model.getState() << '\n';

    Sample true_sim {true_model.simulate(numSims)};
    std::cout << "Fields:\n" << true_model.getFields().transpose() << '\n';
    std::cout << "Couplings:\n" << true_model.getCouplings() << '\n';
    std::cout << '\n';
    std::cout << '\n';

    // inferred model
    const Eigen::MatrixXd J1 {Parameters::getGaussianCouplings(numUnits, 0.0, 1.0)};
    const Eigen::VectorXd h1 {Parameters::getUniformFields(numUnits, -1.3, 1.3)}; 

    // stop taking numUnits... it's un-necessary
    EqModel ml_model {J1, h1};

    const Eigen::MatrixXd J2 {Parameters::getGaussianCouplings(numUnits, 0.0, 1.0)};
    const Eigen::VectorXd h2 {Parameters::getUniformFields(numUnits, -1.3, 1.3)}; 

    EqModel pl_model {J2, h2};
    //NeqModel est_model {numUnits, J, h};

    /*
        INFERENCE
    */
    int maxSteps {1000};
    double lr {0.01};
    int numBurn {1000};
    bool calcLLH {true};

    Timer t;
    Inverse::maxLikelihoodTraj ml_out {Inverse::maxLikelihood(ml_model, true_sim, maxSteps, lr, numSims, numBurn, calcLLH)};  
    std::cout << "ML inference took: " << t.elapsed() << " seconds.\n";
    std::cout << "Fields:\n" << ml_model.getFields().transpose() << '\n';
    std::cout << "Couplings:\n" << ml_model.getCouplings() << '\n';
    
    std::cout << '\n';
    std::cout << '\n';

    std::cout << "LLHs:\n";
    for (double llh : ml_out.LLHs) {
        std::cout << llh << '\n';

    }

    // t.reset();
    // Inverse::maxLikelihoodTraj pl_out {Inverse::maxLikelihood(pl_model, true_sim, maxSteps, lr)};  
    // std::cout << "PL inference took: " << t.elapsed() << " seconds.\n";
    // std::cout << "Fields:\n" << pl_model.getFields().transpose() << '\n';
    // std::cout << "Couplings:\n" << pl_model.getCouplings() << '\n';

    return 0;
}