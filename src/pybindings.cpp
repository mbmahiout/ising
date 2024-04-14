#include "sample.h"
#include "models.h"
#include "exact_infer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(ising, m) {
    // sample
    py::class_<Sample>(m, "Sample")
        // constructor
        .def(py::init<Eigen::MatrixXi>())

        // getters (general properties)
        .def("getNumUnits", &Sample::getNumUnits)
        .def("getNumBins", &Sample::getNumBins)
        .def("getStates", &Sample::getStates)
        .def("getState", &Sample::getState)

        // getters (statistics)
        .def("getMeans", &Sample::getMeans)
        .def("getPairwiseCorrs", &Sample::getPairwiseCorrs)
        .def("getConnectedCorrs", &Sample::getConnectedCorrs)
        .def("getDelayedCorrs", &Sample::getDelayedCorrs);

    // models
    py::class_<EqModel, std::shared_ptr<EqModel>>(m, "EqModel")
            // constructor
            .def(py::init<Eigen::MatrixXd, Eigen::VectorXd>())
            
            // setters
            .def("setFields", &EqModel::setFields)
            .def("setCouplings", &EqModel::setCouplings)

            // getters
            .def("getNumUnits", &EqModel::getNumUnits)
            .def("getFields", &EqModel::getFields)
            .def("getCouplings", &EqModel::getCouplings)
            
            // simulation
            .def("simulate", &EqModel::simulate);

    // inverse
    py::class_<Inverse::maxLikelihoodTraj>(m, "maxLikelihoodTraj")
        .def(py::init<>())
        
        // parameters trajectory
        .def_readwrite("fieldsHistory", &Inverse::maxLikelihoodTraj::fieldsHistory)
        .def_readwrite("couplingsHistory", &Inverse::maxLikelihoodTraj::couplingsHistory)
        
        // for convergence testing
        .def_readwrite("fieldsDiffsEMA", &Inverse::maxLikelihoodTraj::fieldsDiffsEMA)
        .def_readwrite("couplingsDiffsEMA", &Inverse::maxLikelihoodTraj::couplingsDiffsEMA)
        .def_readwrite("fieldsGrads", &Inverse::maxLikelihoodTraj::fieldsGrads)
        .def_readwrite("couplingsGrads", &Inverse::maxLikelihoodTraj::couplingsGrads)

        // optionally (for testing)
        .def_readwrite("LLHs", &Inverse::maxLikelihoodTraj::LLHs);

    m.def("maxLikelihoodEq", 
          &Inverse::maxLikelihood<EqModel>,
          "Gradient ascent for LLH maximization",
          py::arg("model"), 
          py::arg("sample"), 
          py::arg("maxSteps"), 
          py::arg("learningRate") = 0.1,
          py::arg("alpha") = 0.1, 
          py::arg("tolerance") = 1e-5, 
          py::arg("numSims") = 0, 
          py::arg("numBurn") = 0, 
          py::arg("calcLLH") = false);

}