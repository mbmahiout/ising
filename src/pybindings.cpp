#include "sample.h"
#include "models.h"
#include "exact_infer.h"
#include <pybind11/pybind11.h>
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
            .def(py::init<const int, Eigen::MatrixXd, Eigen::VectorXd>())
            
            // setters
            .def("setFields", &EqModel::setFields)
            .def("setCouplings", &EqModel::setCouplings)

            // getters
            .def("getNumUnits", &EqModel::getNumUnits)
            .def("getFields", &EqModel::getFields)
            .def("getCouplings", &EqModel::getCouplings)
            
            // simulation
            .def("simulate", &EqModel::simulate);


//     py::class_<NeqModel, std::shared_ptr<NeqModel>>(m, "NeqModel")
//             .def(py::init<const int, Eigen::MatrixXd, Eigen::VectorXd>())
//             .def("getFields", &NeqModel::getFields)
//             .def("getCouplings", &NeqModel::getCouplings)
//             .def("simulate", &NeqModel::simulate);

    // inverse
    m.def("setMaxLikelihoodParamsEq", 
          &Inverse::setMaxLikelihoodParams<EqModel>,
          "Set maximum likelihood parameters for EqModel",
          py::arg("model"), py::arg("sample"), py::arg("maxSteps"), 
          py::arg("learningRate"), py::arg("numSims"), py::arg("numBurn"));

//     m.def("setMaxLikelihoodParamsNeq", 
//           &Inverse::setMaxLikelihoodParams<NeqModel>,
//           "Set maximum likelihood parameters for NeqModel",
//           py::arg("model"), py::arg("sample"), py::arg("maxSteps"), 
//           py::arg("learningRate"), py::arg("numSims"), py::arg("numBurn"));
}
