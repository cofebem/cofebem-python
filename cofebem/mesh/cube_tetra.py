#!/usr/bin/env python3
"""
graded_box_layers.py
Pure-numpy + meshio script that

  • builds a hexahedral mesh of a box with finer spacing inside a sub-cube
  • returns the connectivity array
  • annotates each hex with an integer 'Layer' = its z-index
"""

import numpy as np
import meshio

# ----------------------------- user knobs ------------------------------
L = 1.0  # half-size of the outer box  ==> domain = [-L, L]^3
l = 0.3  # half-size of fine cube      ==> fine zone = [-l, l]^3
hC = 0.20  # coarse spacing
hF = 0.05  # fine spacing  (must divide hC evenly for conformity)
outfile = "graded_box_with_layers.xdmf"
# ----------------------------------------------------------------------


def make_axis(lo, hi, half_fine, h_coarse, h_fine):
    """Return 1-D coordinate list with fine spacing in |x|<half_fine."""
    # coarse on the left
    left = np.arange(lo, -half_fine, h_coarse)
    # fine in the middle (both ends included)
    mid = np.arange(-half_fine, half_fine + 1e-12, h_fine)
    # coarse on the right
    right = np.arange(half_fine + h_coarse, hi + 1e-12, h_coarse)
    axis = np.unique(np.concatenate((left, mid, right)))
    if axis[-1] < hi - 1e-12:  # be sure to hit hi exactly
        axis = np.append(axis, hi)
    return axis


def structured_hex_connectivity(nx, ny, nz):
    """
    Build connectivity and layer id in one go.

    returns
    -------
    conn  : (Ncells, 8) int64   hexahedron corner indices (VTK order)
    layer : (Ncells,)  int64    z-layer number  k = 0 … nz-2
    """
    I, J, K = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
    )
    base = I + J * nx + K * nx * ny
    conn = np.vstack(
        [
            base,
            base + 1,
            base + 1 + nx,
            base + nx,
            base + nx * ny,
            base + 1 + nx * ny,
            base + 1 + nx + nx * ny,
            base + nx + nx * ny,
        ]
    ).T
    conn = conn.reshape(-1, 8).astype(np.int64)
    layer = K.ravel().astype(np.int64)  # each voxel gets its K index
    return conn, layer


def main():
    # 1-D axes
    x = make_axis(-L, L, l, hC, hF)
    y = x.copy()
    z = x.copy()
    nx, ny, nz = len(x), len(y), len(z)

    # node coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # connectivity + layer id
    hexes, layer = structured_hex_connectivity(nx, ny, nz)

    # meshio export with cell-data
    mesh = meshio.Mesh(
        points,
        [("hexahedron", hexes)],
        cell_data={"Layer": [layer]},
    )
    meshio.write(outfile, mesh)
    print(f"✓ wrote {outfile}   ({points.shape[0]} nodes, {hexes.shape[0]} hexes)")
    print("  Layer range =", layer.min(), "…", layer.max())


if __name__ == "__main__":
    main()
