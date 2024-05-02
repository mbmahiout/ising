#include "sample.h"
#include "models.h"
#include "grad_ascent.h"
#include "utils.h"
#include "timer.h"
#include <iostream>
#include <Eigen/Dense>
#include <string>


template <typename T>
void printParams(const T& model, std::string_view modelName) {
    std::cout << modelName << ":\n";
    std::cout << "h:\n" << model.getFields().transpose() << '\n';
    std::cout << "J:\n" << model.getCouplings() << '\n';
}


int main() {
    // true model
    const int numUnits {3};
    const int numSims {10};

    const Eigen::MatrixXd J {Parameters::getGaussianCouplings(numUnits, 0.0, 1.0)};
    const Eigen::VectorXd h {Parameters::getUniformFields(numUnits, -1.3, 1.3)};

    NeqModel true_model {J, h};

    Sample true_sim {true_model.simulate(numSims)};

    printParams(true_model, "True model");

    std::cout << '\n';

    std::cout << "Simulation:\n" << true_sim.getStates() << '\n';

    /*
        INFERENCE
    */
    // inferred models
    const Eigen::MatrixXd J1 {Parameters::getGaussianCouplings(numUnits, 0.0, 1.0)};
    const Eigen::VectorXd h1 {Parameters::getUniformFields(numUnits, -1.3, 1.3)}; 

    NeqModel ml_model {J1, h1};

    // const Eigen::MatrixXd J2 {Parameters::getGaussianCouplings(numUnits, 0.0, 1.0)};
    // const Eigen::VectorXd h2 {Parameters::getUniformFields(numUnits, -1.3, 1.3)}; 

    // EqModel pl_model {J2, h2};

    int maxSteps {100};
    double lr {0.01};
    int numBurn {1000};
    bool calcLLH {false};
    double alpha {0.1};
    double tol {1e-5};
    bool useAdam {false};

    Timer t;

    /*
        ML INFERENCE
    */

    //Inverse::gradAscOut ml_out {Inverse::gradientAscent(ml_model, true_sim, maxSteps, lr, useAdam)};  
    Inverse::gradAscOut ml_out {Inverse::gradientAscent(ml_model, true_sim, maxSteps, lr, useAdam)};  
    std::cout << "ML inference took: " << t.elapsed() << " seconds.\n";

    std::cout << '\n';

    printParams(ml_model, "ML model");

    std::cout << '\n';

    t.reset();

    /*
        PL INFERENCE
    */

    // Inverse::gradAscOut pl_out {Inverse::gradientAscent(pl_model, true_sim, maxSteps, lr, useAdam)};  
    // std::cout << "PL inference took: " << t.elapsed() << " seconds.\n";
    
    // std::cout << '\n';

    // printParams(pl_model, "PL model");

    // std::cout << '\n';

    return 0;
}