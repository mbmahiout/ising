#include "sample.h"
#include "models.h"
#include "grad_ascent.h"
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

    // gradient ascent
    py::class_<Inverse::paramsHistory>(m, "paramsHistory")
        .def(py::init<>())
        .def_readwrite("fields", &Inverse::paramsHistory::fields)
        .def_readwrite("couplings", &Inverse::paramsHistory::couplings);

    py::class_<Inverse::gradsHistory>(m, "gradsHistory")
        .def(py::init<>())
        .def_readwrite("fieldsGrads", &Inverse::gradsHistory::fieldsGrads)
        .def_readwrite("couplingsGrads", &Inverse::gradsHistory::couplingsGrads);

    py::class_<Inverse::statsHistory>(m, "statsHistory")
        .def(py::init<>())
        .def_readwrite("avFields", &Inverse::statsHistory::avFields)
        .def_readwrite("avCouplings", &Inverse::statsHistory::avCouplings)

        .def_readwrite("sdFields", &Inverse::statsHistory::sdFields)
        .def_readwrite("sdCouplings", &Inverse::statsHistory::sdCouplings)

        .def_readwrite("minFields", &Inverse::statsHistory::minFields)
        .def_readwrite("minCouplings", &Inverse::statsHistory::minCouplings)

        .def_readwrite("maxFields", &Inverse::statsHistory::maxFields)
        .def_readwrite("maxCouplings", &Inverse::statsHistory::maxCouplings)

        .def_readwrite("LLHs", &Inverse::statsHistory::LLHs);

    py::class_<Inverse::gradAscOut>(m, "gradAscOut")
        .def(py::init<>())
        .def_readwrite("params", &Inverse::gradAscOut::params)
        .def_readwrite("grads", &Inverse::gradAscOut::grads)
        .def_readwrite("stats", &Inverse::gradAscOut::stats);

    m.def("gradientAscentEQ", 
          &Inverse::gradientAscent<EqModel>,
          "Gradient ascent for LLH/PL maximization",
          py::arg("model"), 
          py::arg("sample"), 
          py::arg("maxSteps"), 
          py::arg("learningRate") = 0.1,
          py::arg("useAdam") = true,
          py::arg("beta1") = 0.9,
          py::arg("beta2") = 0.999,
          py::arg("epsilon") = 0.1,
          py::arg("winSize") = 10,
          py::arg("tolerance") = 1e-5, 
          py::arg("numSims") = 0, 
          py::arg("numBurn") = 0, 
          py::arg("calcLLH") = false);

}