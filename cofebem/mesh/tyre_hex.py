import numpy as np
import meshio

# def trimmed_ellipse_arc(a, b, ox, oz, theta_cut, ntt=60):
#     t0 = np.linspace(0.0, 2.0 * np.pi, ntt, endpoint=False)
#     mask = ((t0 >= 0.0) & (t0 <= np.pi + theta_cut)) | (
#         (t0 >= 2.0 * np.pi - theta_cut) & (t0 <= 2.0 * np.pi)
#     )
#     t = t0[mask]
#     x = ox + a * np.cos(t)
#     z = oz + b * np.sin(t)
#     return x.astype(np.float64), z.astype(np.float64)


def trimmed_ellipse_arc(a, b, ox, oz, theta_cut, ntt=60):
    t0 = np.linspace(0.0, 2.0 * np.pi, ntt, endpoint=False)

    # Split into the two kept segments
    seg1 = t0[(t0 >= 0.0) & (t0 <= np.pi + theta_cut)]
    seg2 = t0[(t0 >= 2.0 * np.pi - theta_cut) & (t0 < 2.0 * np.pi)]

    # IMPORTANT: order them so that 2π...0 is connected by adjacency
    t = np.concatenate([seg2, seg1])

    x = ox + a * np.cos(t)
    z = oz + b * np.sin(t)
    return x.astype(np.float64), z.astype(np.float64)


def tyre_hex_mesh(
    a=0.20,
    b=0.10,
    t=0.03,
    ox=0.0,
    oz=0.5,
    theta_cut=np.pi / 6,
    nr=3,
    ntt=40,
    npp=80,
    filename="tyre_hex.vtk",
    periodic_phi=True,
    eps=1e-8,
):
    if nr < 2:
        raise ValueError("nr must be >= 2")
    if ntt < 2:
        raise ValueError("ntt must be >= 2")
    if npp < 3:
        raise ValueError("np must be >= 3")

    a_outer = a + t
    b_outer = b + t
    if a_outer <= 0 or b_outer <= 0:
        raise ValueError(
            "t makes outer semi-axes non-positive. Choose smaller negative t or positive t."
        )

    rr = np.linspace(0.0, 1.0, nr)

    x_layers = []
    z_layers = []
    for r in rr:
        ai = (1.0 - r) * a + r * a_outer
        bi = (1.0 - r) * b + r * b_outer
        xk, zk = trimmed_ellipse_arc(ai, bi, ox, oz, theta_cut, ntt=ntt)
        x_layers.append(xk)
        z_layers.append(zk)

    ns = len(x_layers[0])
    if ns < 2:
        raise ValueError(
            "Trim produced too few points; increase ntt or reduce theta_cut."
        )
    for k in range(nr):
        if len(x_layers[k]) != ns:
            raise RuntimeError(
                "Unexpected: layer point counts differ. (Should not happen with same ntt/theta_cut)"
            )

    phi = np.linspace(0.0, 2.0 * np.pi, npp, endpoint=True)
    ct = np.cos(phi)
    st = np.sin(phi)

    z_min = min(np.min(z) for z in z_layers)
    if z_min < eps:
        shift = eps - z_min
        z_layers = [z + shift for z in z_layers]

    # global id: (ir, is, ip) -> index
    def gid(ir, is_, ip):
        return (ir * ns + is_) * npp + ip

    points = np.zeros((nr * ns * npp, 3), dtype=np.float64)
    for ir in range(nr):
        x = x_layers[ir]
        r = z_layers[ir]

        X = x[:, None]
        Y = -r[:, None] * st[None, :]
        Z = r[:, None] * ct[None, :]

        for is_ in range(ns):
            base = gid(ir, is_, 0)
            points[base : base + npp, 0] = X[is_, :]
            points[base : base + npp, 1] = Y[is_, :]
            points[base : base + npp, 2] = Z[is_, :]

    hexes = []
    phi_cells = npp if periodic_phi else (npp - 1)

    for ir in range(nr - 1):  # radial cells
        for is_ in range(ns - 1):  #  along-section cells (open in section direction)
            for ip in range(phi_cells):
                ip1 = (ip + 1) % npp

                n000 = gid(ir, is_, ip)
                n001 = gid(ir, is_, ip1)
                n010 = gid(ir, is_ + 1, ip)
                n011 = gid(ir, is_ + 1, ip1)

                n100 = gid(ir + 1, is_, ip)
                n101 = gid(ir + 1, is_, ip1)
                n110 = gid(ir + 1, is_ + 1, ip)
                n111 = gid(ir + 1, is_ + 1, ip1)

                hexes.append([n000, n001, n011, n010, n100, n101, n111, n110])

    mesh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", np.asarray(hexes, dtype=np.int64))],
    )
    mesh.write(filename)

    print(f"Wrote: {filename}")
    print(f"  points = {points.shape[0]}")
    print(f"  hexes  = {len(hexes)}")
    print(f"  dims   = nr={nr}, ns={ns} (from trimmed arc), npp={npp}")
    print(
        f"  params = a={a}, b={b}, t={t}, ox={ox}, oz={oz}, theta_cut={theta_cut}, ntt={ntt}"
    )


# if __name__ == "__main__":
#     tyre_hex_mesh(
#         a=0.20,
#         b=0.10,
#         t=0.03,
#         ox=0.0,
#         oz=0.5,
#         theta_cut=np.pi / 6,
#         nr=3,
#         ntt=40,
#         npp=80,
#         filename="tyre_hex.xdmf",
#         periodic_phi=True,
#     )
