#include "yourpkg/constrained_cg.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

void test_basic_convergence() {
    size_t n = 10;
    std::vector<double> K(n * n);
    std::vector<double> gap(n);
    
    // Create a simple positive definite matrix
    for (size_t i = 0; i < n; ++i) {
        K[i * n + i] = 10.0;
        for (size_t j = i + 1; j < n; ++j) {
            K[i * n + j] = K[j * n + i] = 0.1;
        }
        gap[i] = -0.01 * i;
    }
    
    auto result = yourpkg::ConstrainedCG::solve(
        K.data(), n, "mix", gap.data(), 1000, 1e-6
    );
    
    assert(result.iterations > 0);
    assert(result.pressure.size() == n);
    assert(result.displacement.size() == n);
    
    std::cout << "✓ Basic convergence test passed\n";
}

int main() {
    test_basic_convergence();
    std::cout << "All C++ tests passed!\n";
    return 0;
}