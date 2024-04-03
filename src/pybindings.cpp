#include "sample.h"
#include "models.h"
#include "inverse.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(ising, m) {
    // sample
    py::class_<Sample>(m, "Sample")
        .def(py::init<Eigen::MatrixXi>())
        .def("getStates", &Sample::getStates)
        .def("getMeans", &Sample::getMeans)
        .def("getPairwiseCorrs", &Sample::getPairwiseCorrs)
        .def("getConnectedCorrs", &Sample::getConnectedCorrs)
        .def("getDelayedCorrs", &Sample::getDelayedCorrs);

    // models
    py::class_<EqModel, std::shared_ptr<EqModel>>(m, "EqModel")
            .def(py::init<const int, Eigen::MatrixXd, Eigen::VectorXd>())
            .def("getFields", &EqModel::getFields)
            .def("getCouplings", &EqModel::getCouplings)
            .def("simulate", &EqModel::simulate);

    py::class_<NeqModel, std::shared_ptr<NeqModel>>(m, "NeqModel")
            .def(py::init<const int, Eigen::MatrixXd, Eigen::VectorXd>())
            .def("getFields", &NeqModel::getFields)
            .def("getCouplings", &NeqModel::getCouplings)
            .def("simulate", &NeqModel::simulate);

    // inverse
    m.def("setMaxLikelihoodParamsEqModel", 
          &Inverse::setMaxLikelihoodParams<EqModel>,
          "Set maximum likelihood parameters for EqModel",
          py::arg("model"), py::arg("sample"), py::arg("maxSteps"), 
          py::arg("learningRate"), py::arg("numSims"), py::arg("numBurn"));

}
