import numpy as np

# ---- load npz ----
path = "../out_elasticity/FlexData_21x21.npz"   # adjust if needed
z = np.load(path)

K = z["K"]                    # you saved K.T, so undo it
m = np.asarray(z["M"], float)   # lumped boundary mass (vector)
fac_cent = z["facet_centers"]   # (m_facets, 3)
nod_ids  = z["boundary_dofs"]   # global dof ids of boundary P1 nodes
nod_xyz  = z["boundary_coords"] # (n_nodes, 3)

# ---- 1) basic shape checks ----
n_nodes, n_facets = K.shape
assert m.shape == (n_nodes,), f"m shape {m.shape} != ({n_nodes},)"
print(f"K: {K.shape}, m: {m.shape}, facets: {fac_cent.shape[0]}, nodes: {n_nodes}")

# ---- 2) positivity and support on top surface ----
assert np.all(m >= -1e-14), f"Negative entries in m: min={m.min()}"
zvals = nod_xyz[:, 2]
print("z-range on boundary nodes:", zvals.min(), zvals.max())
assert np.allclose(zvals, 1.0, atol=1e-5), "Boundary nodes are not on z≈1 surface"

# ---- 3) area and centroid (unit square -> area=1, centroid at (0.5,0.5)) ----
area = m.sum()
cx = np.dot(m, nod_xyz[:, 0])  # ∫ x ds ≈ sum_i m_i x_i
cy = np.dot(m, nod_xyz[:, 1])  # ∫ y ds ≈ sum_i m_i y_i
print(f"area ≈ {area:.12f} (target 1.0)")
print(f"centroid ≈ ({cx:.8f}, {cy:.8f}) (target ~0.5, 0.5)")
assert abs(area - 1.0) < 5e-8, "Sum(m) should equal surface area ≈ 1.0"
assert abs(cx - 0.5) < 2e-2 and abs(cy - 0.5) < 2e-2, "Centroid off; check facet tagging or m assembly"

# ---- 4) symmetry / SPD sanity for H = K^T diag(m) K ----
# (use a few random Rayleigh quotients instead of a full eigendecomp)
H = K.T @ (m[:, None] * K)
sym_err = np.linalg.norm(H - H.T, ord='fro') / (np.linalg.norm(H, ord='fro') + 1e-30)
print(f"symmetry rel. error ||H-H^T||/||H|| = {sym_err:.3e}")
assert sym_err < 1e-12, "H not symmetric; check K orientation"

rng = np.random.default_rng(0)
for t in range(5):
    v = rng.standard_normal(n_facets)
    q = v @ (H @ v)
    assert q >= -1e-10 * (np.linalg.norm(H) * np.linalg.norm(v)**2 + 1.0), "H not PSD"
print("H appears PSD by random tests.")

# ---- 5) quick physical consistency checks ----
# uniform pressure -> displacement should be smooth and positive (relative scale)
p_uni = np.ones(n_facets)
u = K @ p_uni
u_wavg = np.dot(m, u) / area
print(f"weighted avg displacement for p=1: {u_wavg:.6e} (arbitrary units, must be finite and >0)")

# random nonnegative pressure -> no NaNs, finite values
p_rand = np.abs(rng.standard_normal(n_facets))
u_rand = K @ p_rand
assert np.isfinite(u_rand).all(), "Non-finite displacements from K @ p"
print("All checks passed.")
