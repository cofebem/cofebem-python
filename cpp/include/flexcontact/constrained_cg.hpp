// cpp/include/flexcontact/constrained_cg.hpp
// Header file for Constrained Conjugate Gradient solver

#pragma once

#include <Eigen/Dense>
#include <string>

namespace flexcontact {

/**
 * @brief Result structure for the Constrained CG solver
 */
struct CCGResult {
    Eigen::VectorXd pressure;        ///< Contact pressure solution (size n)
    Eigen::VectorXd displacement;    ///< Displacement field (size n)
    Eigen::MatrixXd error_history;   ///< Error history (iterations × 3)
                                     ///< Columns: [displ_error, error, orthogonality]
    int iterations;                  ///< Number of iterations performed
};

/**
 * @brief Constrained Conjugate Gradient solver for contact problems
 * 
 * This class implements a constrained conjugate gradient method for solving
 * contact mechanics problems with inequality constraints (non-penetration).
 * 
 * The solver finds pressure distribution p that satisfies:
 * - Non-negativity: p >= 0
 * - Complementarity: p * (u + gap) = 0
 * where u = K * p is the displacement.
 */
class ConstrainedCG {
public:
    /**
     * @brief Solve contact problem using Constrained Conjugate Gradient
     * 
     * @param K Stiffness/flexibility matrix (n×n, symmetric positive definite)
     * @param error_type Error criterion for convergence:
     *                   - "displacement": norm of residual at active contacts
     *                   - "mix": geometric mean of displacement error and orthogonality
     *                   - "nw": relative change in total residual norm
     * @param gap Gap values (size n). Positive gap = separation, negative = penetration
     * @param max_iter Maximum number of iterations
     * @param tolerance Convergence tolerance for the error criterion
     * @param pressure_factor Initial pressure scaling factor (default: 1e12)
     * @param initial_pressure Optional initial pressure guess (nullptr for default)
     * 
     * @return CCGResult containing pressure, displacement, and convergence history
     * 
     * @throws std::runtime_error if inputs are invalid or solver fails
     * 
     * @note Algorithm based on constrained conjugate gradient with active set updates
     * @note Matrix K should be positive definite for guaranteed convergence
     * 
     * @example
     * @code
     * Eigen::MatrixXd K(100, 100);
     * Eigen::VectorXd gap(100);
     * // ... initialize K and gap ...
     * 
     * auto result = ConstrainedCG::solve(K, "mix", gap, 1000, 1e-6);
     * std::cout << "Converged in " << result.iterations << " iterations\n";
     * @endcode
     */
    static CCGResult solve(
        const Eigen::MatrixXd& K,
        const std::string& error_type,
        const Eigen::VectorXd& gap,
        int max_iter,
        double tolerance,
        double pressure_factor = 1e12,
        const Eigen::VectorXd* initial_pressure = nullptr
    );
    
private:
    // Helper functions can be added here if needed for internal use
};

} // namespace flexcontact