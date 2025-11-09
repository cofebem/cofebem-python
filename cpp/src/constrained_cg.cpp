// cpp/src/constrained_cg.cpp
// Complete implementation of Constrained Conjugate Gradient with Eigen

#include "flexcontact/constrained_cg.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace flexcontact {

CCGResult ConstrainedCG::solve(
    const Eigen::MatrixXd& K,
    const std::string& error_type,
    const Eigen::VectorXd& gap,
    int max_iter,
    double tolerance,
    double pressure_factor,
    const Eigen::VectorXd* initial_pressure
) {
    // =========================================================================
    // Input validation
    // =========================================================================
    if (K.rows() != K.cols()) {
        throw std::runtime_error("K must be a square matrix");
    }
    
    const int n = K.rows();
    
    if (gap.size() != n) {
        throw std::runtime_error("gap size must match K dimensions");
    }
    
    if (error_type != "displacement" && error_type != "mix" && error_type != "nw") {
        throw std::runtime_error("error_type must be 'displacement', 'mix', or 'nw'");
    }
    
    // =========================================================================
    // Initialize variables
    // =========================================================================
    Eigen::VectorXd p(n);      // Pressure
    Eigen::VectorXd ub(n);     // Upper bound = -gap
    Eigen::VectorXd w(n);      // Residual/displacement
    Eigen::VectorXd t(n);      // Search direction
    Eigen::VectorXd t_(n);     // Previous search direction
    Eigen::VectorXd q(n);      // K * t
    
    // ub = -gap
    ub = -gap;
    
    // =========================================================================
    // Initialize pressure
    // =========================================================================
    // Python equivalent: p = np.maximum(-gap, 0) * pressure_factor
    if (initial_pressure != nullptr) {
        // Warm start provided (but we still use the standard initialization)
        p = (-gap).cwiseMax(0.0) * pressure_factor;
    } else {
        // Cold start: p = max(-gap, 0) * pressure_factor
        p = (-gap).cwiseMax(0.0) * pressure_factor;
    }
    
    // =========================================================================
    // Initial residual: w = K * p - ub
    // =========================================================================
    w = K * p - ub;
    
    // =========================================================================
    // Initial search direction: t = w
    // =========================================================================
    t = w;
    t_.setZero();
    
    // =========================================================================
    // Iteration variables
    // =========================================================================
    double d = 0.0;
    double error = 1.0;
    double error_ = 1.0;
    
    // Pre-allocate error history matrix
    Eigen::MatrixXd error_history(max_iter, 3);
    
    // =========================================================================
    // Main CG iteration loop
    // =========================================================================
    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        
        // =====================================================================
        // Update search direction (if not first iteration)
        // =====================================================================
        if (iter > 0) {
            double ratio = d * error / error_;
            
            // Python: t[p>0] = w[p>0] + ratio * t_[p>0]
            //         t[p<=0] = 0
            for (int i = 0; i < n; ++i) {
                if (p(i) > 0.0) {
                    t(i) = w(i) + ratio * t_(i);
                } else {
                    t(i) = 0.0;
                }
            }
        }
        
        // =====================================================================
        // Compute q = K * t
        // =====================================================================
        q = K * t;
        
        // =====================================================================
        // Compute step size tau
        // =====================================================================
        // tau = dot(w, t) / dot(t, q)
        double wt = w.dot(t);
        double tq = t.dot(q);
        
        if (std::abs(tq) < 1e-16) {
            // Avoid division by zero
            throw std::runtime_error("Division by zero in tau computation");
        }
        
        double tau = wt / tq;
        
        // =====================================================================
        // Update pressure: p = p - tau * t, then p = max(p, 0)
        // =====================================================================
        p = (p - tau * t).cwiseMax(0.0);
        
        // =====================================================================
        // Find set I: intersection of zero pressure and penetration
        // =====================================================================
        // Python: zero_pressure = np.where(p == 0)[0]
        //         penetration = np.where(w < 0)[0]
        //         set_I = np.intersect1d(zero_pressure, penetration)
        
        std::vector<int> set_I;
        set_I.reserve(n);  // Pre-allocate for efficiency
        
        for (int i = 0; i < n; ++i) {
            if (p(i) == 0.0 && w(i) < 0.0) {
                set_I.push_back(i);
            }
        }
        
        // =====================================================================
        // Update d flag and apply correction to set I
        // =====================================================================
        if (set_I.empty()) {
            d = 1.0;
        } else {
            d = 0.0;
            // Python: p[set_I] -= tau * w[set_I]
            for (int i : set_I) {
                p(i) -= tau * w(i);
            }
        }
        
        // =====================================================================
        // Store previous search direction: t_ = t
        // =====================================================================
        t_ = t;
        
        // =====================================================================
        // Update residual: w = K * p - ub
        // =====================================================================
        w = K * p - ub;
        
        // =====================================================================
        // Compute norms and errors
        // =====================================================================
        double nw = w.norm();  // L2 norm of w
        
        // Save previous error
        error_ = error;
        
        // ---------------------------------------------------------------------
        // Compute displacement error: norm of w where p > 0
        // ---------------------------------------------------------------------
        // Python: displ_error = np.linalg.norm(w[p>0], 2) / nw
        double sum_displ_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            if (p(i) > 0.0) {
                sum_displ_sq += w(i) * w(i);
            }
        }
        double displ_error = std::sqrt(sum_displ_sq) / (nw + 1e-16);
        
        // ---------------------------------------------------------------------
        // Compute orthogonality
        // ---------------------------------------------------------------------
        // Python: ort = np.abs(np.dot(w, p) / nw)
        double wp = w.dot(p);
        double ort = std::abs(wp) / (nw + 1e-16);
        
        // =====================================================================
        // Determine error based on error_type
        // =====================================================================
        if (error_type == "displacement") {
            error = displ_error;
            
        } else if (error_type == "mix") {
            error = std::sqrt(displ_error * ort);
            
        } else if (error_type == "nw") {
            error = nw;
            
            // Special check for "nw" mode
            if (std::abs((error - error_) / (error_ + 1e-16)) < tolerance) {
                // Store final error history
                error_history(iter, 0) = displ_error;
                error_history(iter, 1) = std::abs((error - error_) / (error_ + 1e-16));
                error_history(iter, 2) = ort;
                
                // Prepare result and return early
                CCGResult result;
                result.pressure = p;
                result.displacement = K * p;
                result.error_history = error_history.topRows(iter + 1);
                result.iterations = iter + 1;
                return result;
            }
        }
        
        // =====================================================================
        // Store error history for this iteration
        // =====================================================================
        error_history(iter, 0) = displ_error;
        error_history(iter, 1) = error;
        error_history(iter, 2) = ort;
        
        // =====================================================================
        // Check convergence
        // =====================================================================
        if (error < tolerance) {
            break;  // Converged!
        }
    }
    
    // =========================================================================
    // Prepare final result
    // =========================================================================
    CCGResult result;
    result.pressure = p;
    result.displacement = K * p;
    result.error_history = error_history.topRows(std::min(iter + 1, max_iter));
    result.iterations = iter + 1;
    
    return result;
}

} // namespace flexcontact