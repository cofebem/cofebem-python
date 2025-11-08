// constrained_cg.cpp
// Efficient C++ implementation of Constrained Conjugate Gradient solver
// Compile with: c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) constrained_cg.cpp -o constrained_cg$(python3-config --extension-suffix)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

namespace py = pybind11;

// BLAS-like operations for better performance
inline double dot_product(const double* x, const double* y, size_t n) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

inline double norm_l2(const double* x, size_t n) {
    return std::sqrt(dot_product(x, x, n));
}

// Matrix-vector multiplication: result = K * vec
inline void matvec(const double* K, const double* vec, double* result, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t j = 0; j < n; ++j) {
            sum += K[i * n + j] * vec[j];
        }
        result[i] = sum;
    }
}

struct CCGResult {
    py::array_t<double> pressure;
    py::array_t<double> displacement;
    py::array_t<double> error_history;
    int iterations;
};

CCGResult constrained_CG(
    py::array_t<double> K_array,
    const std::string& error_type,
    py::array_t<double> coord,
    py::array_t<double> dofs,
    py::array_t<double> gap,
    int max_iter,
    double tolerance,
    double pressure_factor = 1e12,
    py::object initial_pressure_obj = py::none()
) {
    // Get array buffers
    auto K_buf = K_array.request();
    auto gap_buf = gap.request();
    
    if (K_buf.ndim != 2 || K_buf.shape[0] != K_buf.shape[1]) {
        throw std::runtime_error("K must be a square matrix");
    }
    
    size_t n = K_buf.shape[0];
    
    if (gap_buf.size != static_cast<ssize_t>(n)) {
        throw std::runtime_error("gap size must match K dimensions");
    }
    
    const double* K = static_cast<double*>(K_buf.ptr);
    const double* gap_ptr = static_cast<double*>(gap_buf.ptr);
    
    // Allocate working arrays
    std::vector<double> p(n);
    std::vector<double> ub(n);
    std::vector<double> w(n);
    std::vector<double> t(n);
    std::vector<double> t_(n);
    std::vector<double> q(n);
    std::vector<double> temp(n);
    
    // Initialize ub = -gap
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        ub[i] = -gap_ptr[i];
    }
    
    // Initialize pressure
    if (!initial_pressure_obj.is_none()) {
        auto init_p = initial_pressure_obj.cast<py::array_t<double>>();
        auto init_p_buf = init_p.request();
        const double* init_p_ptr = static_cast<double*>(init_p_buf.ptr);
        
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            p[i] = std::max(-gap_ptr[i], 0.0) * pressure_factor;
        }
    } else {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            p[i] = std::max(-gap_ptr[i], 0.0) * pressure_factor;
        }
    }
    
    // w = K * p - ub
    matvec(K, p.data(), w.data(), n);
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        w[i] -= ub[i];
    }
    
    // t = w
    std::copy(w.begin(), w.end(), t.begin());
    std::fill(t_.begin(), t_.end(), 0.0);
    
    double d = 0.0;
    double error = 1.0;
    double error_ = 1.0;
    
    // Error history: [displ_error, error, ort]
    std::vector<double> error_history(max_iter * 3);
    
    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        if (iter > 0) {
            double ratio = d * error / error_;
            #pragma omp simd
            for (size_t i = 0; i < n; ++i) {
                if (p[i] > 0) {
                    t[i] = w[i] + ratio * t_[i];
                } else {
                    t[i] = 0.0;
                }
            }
        }
        
        // q = K * t
        matvec(K, t.data(), q.data(), n);
        
        // tau = dot(w, t) / dot(t, q)
        double wt = dot_product(w.data(), t.data(), n);
        double tq = dot_product(t.data(), q.data(), n);
        double tau = wt / tq;
        
        // p = p - tau * t
        // p = max(p, 0)
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            p[i] = std::max(p[i] - tau * t[i], 0.0);
        }
        
        // Find set I: intersection of zero_pressure and penetration
        std::vector<size_t> set_I;
        for (size_t i = 0; i < n; ++i) {
            if (p[i] == 0.0 && w[i] < 0.0) {
                set_I.push_back(i);
            }
        }
        
        if (set_I.empty()) {
            d = 1.0;
        } else {
            d = 0.0;
            for (size_t i : set_I) {
                p[i] -= tau * w[i];
            }
        }
        
        // t_ = t
        std::copy(t.begin(), t.end(), t_.begin());
        
        // w = K * p - ub
        matvec(K, p.data(), w.data(), n);
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            w[i] -= ub[i];
        }
        
        double nw = norm_l2(w.data(), n);
        
        error_ = error;
        
        // Compute displ_error: norm of w where p > 0
        double sum_displ = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (p[i] > 0) {
                sum_displ += w[i] * w[i];
            }
        }
        double displ_error = std::sqrt(sum_displ) / nw;
        
        // Compute orthogonality
        double wp = dot_product(w.data(), p.data(), n);
        double ort = std::abs(wp / nw);
        
        // Determine error based on error_type
        if (error_type == "displacement") {
            error = displ_error;
        } else if (error_type == "mix") {
            error = std::sqrt(displ_error * ort);
        } else if (error_type == "nw") {
            error = nw;
            if (std::abs((error - error_) / error_) < tolerance) {
                error_history[iter * 3] = displ_error;
                error_history[iter * 3 + 1] = std::abs((error - error_) / error_);
                error_history[iter * 3 + 2] = ort;
                
                // Prepare return values
                CCGResult result;
                result.iterations = iter + 1;
                
                // Create numpy arrays
                result.pressure = py::array_t<double>(n);
                auto p_buf = result.pressure.request();
                std::copy(p.begin(), p.end(), static_cast<double*>(p_buf.ptr));
                
                // displacement = K * p
                result.displacement = py::array_t<double>(n);
                auto disp_buf = result.displacement.request();
                matvec(K, p.data(), static_cast<double*>(disp_buf.ptr), n);
                
                // error_history
                result.error_history = py::array_t<double>({iter + 1, 3});
                auto err_buf = result.error_history.request();
                std::copy(error_history.begin(), error_history.begin() + (iter + 1) * 3,
                         static_cast<double*>(err_buf.ptr));
                
                return result;
            }
        }
        
        // Store error history
        error_history[iter * 3] = displ_error;
        error_history[iter * 3 + 1] = error;
        error_history[iter * 3 + 2] = ort;
        
        if (error < tolerance) {
            break;
        }
    }
    
    // Prepare return values
    CCGResult result;
    result.iterations = iter + 1;
    
    result.pressure = py::array_t<double>(n);
    auto p_buf = result.pressure.request();
    std::copy(p.begin(), p.end(), static_cast<double*>(p_buf.ptr));
    
    result.displacement = py::array_t<double>(n);
    auto disp_buf = result.displacement.request();
    matvec(K, p.data(), static_cast<double*>(disp_buf.ptr), n);
    
    result.error_history = py::array_t<double>({iter + 1, 3});
    auto err_buf = result.error_history.request();
    std::copy(error_history.begin(), error_history.begin() + (iter + 1) * 3,
             static_cast<double*>(err_buf.ptr));
    
    return result;
}

PYBIND11_MODULE(constrained_cg, m) {
    m.doc() = "Constrained Conjugate Gradient solver in C++";
    
    py::class_<CCGResult>(m, "CCGResult")
        .def_readonly("pressure", &CCGResult::pressure)
        .def_readonly("displacement", &CCGResult::displacement)
        .def_readonly("error_history", &CCGResult::error_history)
        .def_readonly("iterations", &CCGResult::iterations);
    
    m.def("constrained_CG", &constrained_CG,
          py::arg("K"),
          py::arg("error_type"),
          py::arg("coord"),
          py::arg("dofs"),
          py::arg("gap"),
          py::arg("max_iter"),
          py::arg("tolerance"),
          py::arg("pressure_factor") = 1e12,
          py::arg("initial_pressure") = py::none(),
          "Constrained Conjugate Gradient solver for contact problems");
}