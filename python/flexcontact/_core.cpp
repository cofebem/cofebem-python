#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "flexcontact/constrained_cg.hpp"

namespace py = pybind11;

// Wrapper that converts between numpy arrays and C++ pointers
struct PyCCGResult {
    py::array_t<double> pressure;
    py::array_t<double> displacement;
    py::array_t<double> error_history;
    int iterations;
};

PyCCGResult constrained_cg_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> K_array,
    const std::string& error_type,
    py::array_t<double, py::array::c_style | py::array::forcecast> gap,
    int max_iter,
    double tolerance,
    double pressure_factor = 1e12,
    py::object initial_pressure_obj = py::none()
) {
    auto K_buf = K_array.request();
    auto gap_buf = gap.request();
    
    if (K_buf.ndim != 2 || K_buf.shape[0] != K_buf.shape[1]) {
        throw std::runtime_error("K must be a square matrix");
    }
    if (gap_buf.ndim != 1 || gap_buf.shape[0] != K_buf.shape[0]) {
        throw std::runtime_error("gap must be a vector matching K dimensions");
    }
    
    const int n = static_cast<int>(K_buf.shape[0]);

    // Map numpy arrays to Eigen structures (copy to column-major matrices)
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> K_row_major(
        static_cast<double*>(K_buf.ptr),
        n,
        n
    );
    Eigen::MatrixXd K_mat = K_row_major;

    const Eigen::Map<const Eigen::VectorXd> gap_vec(static_cast<double*>(gap_buf.ptr), n);
    
    Eigen::VectorXd init_pressure_vec;
    const Eigen::VectorXd* init_pressure_ptr = nullptr;
    if (!initial_pressure_obj.is_none()) {
        auto init_arr = initial_pressure_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto init_buf = init_arr.request();
        if (init_buf.ndim != 1 || init_buf.shape[0] != K_buf.shape[0]) {
            throw std::runtime_error("initial_pressure must be a vector matching K dimensions");
        }
        init_pressure_vec = Eigen::Map<const Eigen::VectorXd>(static_cast<double*>(init_buf.ptr), n);
        init_pressure_ptr = &init_pressure_vec;
    }
    
    // Call C++ solver
    auto result = flexcontact::ConstrainedCG::solve(
        K_mat,
        error_type,
        gap_vec,
        max_iter,
        tolerance,
        pressure_factor,
        init_pressure_ptr
    );
    
    // Convert to numpy arrays
    PyCCGResult py_result;
    py_result.iterations = result.iterations;
    
    py_result.pressure = py::array_t<double>(result.pressure.size());
    std::copy(result.pressure.data(),
              result.pressure.data() + result.pressure.size(),
              py_result.pressure.mutable_data());
    
    py_result.displacement = py::array_t<double>(result.displacement.size());
    std::copy(result.displacement.data(),
              result.displacement.data() + result.displacement.size(),
              py_result.displacement.mutable_data());
    
    const auto rows = static_cast<py::ssize_t>(result.error_history.rows());
    const auto cols = static_cast<py::ssize_t>(result.error_history.cols());
    py_result.error_history = py::array_t<double>({rows, cols});
    std::copy(result.error_history.data(),
              result.error_history.data() + (rows * cols),
              py_result.error_history.mutable_data());
    
    return py_result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Core C++ implementations";
    
    py::class_<PyCCGResult>(m, "CCGResult")
        .def_readonly("pressure", &PyCCGResult::pressure)
        .def_readonly("displacement", &PyCCGResult::displacement)
        .def_readonly("error_history", &PyCCGResult::error_history)
        .def_readonly("iterations", &PyCCGResult::iterations);
    
    m.def("constrained_cg", &constrained_cg_wrapper,
          py::arg("K"),
          py::arg("error_type"),
          py::arg("gap"),
          py::arg("max_iter"),
          py::arg("tolerance"),
          py::arg("pressure_factor") = 1e12,
          py::arg("initial_pressure") = py::none(),
          "Constrained Conjugate Gradient solver");
    
}