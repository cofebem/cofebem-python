import numpy as np
import pytest
from flexcontact import constrained_cg

def test_basic_convergence():
    """Test that solver converges on a simple problem."""
    n = 100
    K = np.random.rand(n, n)
    K = (K + K.T) / 2
    K += np.eye(n) * 10
    
    gap = np.random.randn(n) * 0.01
    coord = np.random.rand(n, 3)
    dofs = np.arange(n)
    
    pressure, displacement, error_history = constrained_cg(
        K, "mix", coord, dofs, gap, 1000, 1e-6
    )
    
    assert pressure.shape == (n,)
    assert displacement.shape == (n,)
    assert error_history.shape[1] == 3
    assert error_history[-1, 1] < 1e-6  # Final error < tolerance

def test_contact_constraints():
    """Test that contact constraints are satisfied."""
    n = 50
    K = np.eye(n) * 10
    gap = np.array([0.01 * i for i in range(n)])  # No penetration
    coord = np.random.rand(n, 3)
    dofs = np.arange(n)
    
    pressure, displacement, _ = constrained_cg(
        K, "displacement", coord, dofs, gap, 1000, 1e-6
    )
    
    # Check non-negativity of pressure
    assert np.all(pressure >= -1e-10)
    
    # Check complementarity: p_i * (u_i + gap_i) ≈ 0
    complementarity = pressure * (displacement - gap)
    assert np.allclose(complementarity, 0, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])