import numpy as np
import meshio
import collada


def dae_polylist_to_vtk(dae_path: str, vtp_path: str, dedup_tol: float = 0.0) -> None:
    dae = collada.Collada(dae_path)

    points_out = []

    for geom in dae.geometries:
        for prim in geom.primitives:
            if prim.__class__.__name__ != "Polylist":
                continue

            V = np.asarray(prim.vertex, dtype=np.float64)

            vcounts = np.asarray(prim.vcounts, dtype=np.int64)
            vind = np.asarray(prim.vertex_index, dtype=np.int64)

            cursor = 0
            for k in vcounts:
                poly = vind[cursor : cursor + k]
                cursor += k
                if k < 3:
                    continue

                v0 = poly[0]
                for i in range(1, k - 1):
                    tri = np.array([v0, poly[i], poly[i + 1]], dtype=np.int64)
                    points_out.append(V[tri])  # (3,3)

    if not points_out:
        raise RuntimeError("No triangles produced from Polylist.")

    # Flatten: (ntri,3,3) -> (ntri*3,3)
    pts = np.asarray(points_out, dtype=np.float64).reshape(-1, 3)

    if pts.shape[0] % 3 != 0:
        raise RuntimeError(
            f"Internal error: pts rows not multiple of 3: {pts.shape[0]}"
        )

    # Connectivity is deterministic from ordering
    tris = np.arange(pts.shape[0], dtype=np.int64).reshape(-1, 3)

    if not np.isfinite(pts).all():
        bad = np.where(~np.isfinite(pts).all(axis=1))[0][:10]
        raise RuntimeError(f"Found NaN/Inf coordinates at rows: {bad}")

    # ---- Deduplication (optional, recommended)
    # if dedup_tol > 0.0:
    #     q = np.round(pts / dedup_tol).astype(np.int64)
    #     _, inv, idx = np.unique(q, axis=0, return_inverse=True, return_index=True)
    #     points = pts[idx]
    #     conn = inv[tris]
    # else:
    points, inv = np.unique(pts, axis=0, return_inverse=True)
    conn = inv[tris]

    mesh = meshio.Mesh(points=points, cells=[("triangle", conn.astype(np.int64))])
    meshio.write(vtp_path, mesh)

    print(f"Wrote: {vtp_path}")
    print(f"points={len(points):,}, triangles={conn.shape[0]:,}")


if __name__ == "__main__":
    dae_polylist_to_vtk("./msh_files/model.dae", "dragon.vtk", dedup_tol=1e-8)
