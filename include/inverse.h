#ifndef INVERSE_H
#define INVERSE_H

#include "models.h"
#include <Eigen/Dense>



#endif //INVERSE_H


/*
    idea:
    Instead of having IsingFitter be a class, let's have it take an Ising model (maybe using templates) 
    as a parameter. In this case, we can implementet it as two namespaces EqInverseMethods and 
    NeqInverseMethods. 

    Regarding how to acheive the polymorphism we made use of in the Python implementation, we can use
    function overloading : the verison of updateParameters() that takes an EqModel should implement the
    Boltzmann learning logic, whilst the version of updateParameters() that takes a NeqModel should 
    implement its logic (basically, pseudolikelihood for the EqModel). 
    
    Pseudo-log likelihood for the EqFitter can also be implemented by overlaoding updateParameters()
    basically, when called with:
        1. EqModel, states, numSims, numBurn -> do Boltzmann learning
        2. EqModel, states                   -> do pseudolikelihood maximization
        3. NeqModel, states                  -> do likelihood maximization for NeqModel
*/