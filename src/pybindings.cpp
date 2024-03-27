#include "models.h"
#include "observables.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


namespace py = pybind11;

PYBIND11_MODULE(ising, m) {
    // models
    py::class_<EqModel, std::shared_ptr<EqModel>>(m, "EqModel")
            .def(py::init<const int, Eigen::MatrixXd, Eigen::VectorXd>())
            .def("simulate", &EqModel::simulate);

    py::class_<NeqModel, std::shared_ptr<NeqModel>>(m, "NeqModel")
            .def(py::init<const int, Eigen::MatrixXd, Eigen::VectorXd>())
            .def("simulate", &NeqModel::simulate);

    // observables
    m.def("getPairwiseCorrs", &getPairwiseCorrs, "Calculate pairwise correlations");
    m.def("getMeans", &getMeans, "Calculate means");
}
