import numpy as np


class Sphere:
    def __init__(self, center, radius=1.0):
        self.center = np.array(center)
        self.radius = radius

    def gap(self, pts):

        if pts.ndim == 1:
            pts = pts[np.newaxis, :]

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        x0, y0, z0 = self.center
        R = float(self.radius)

        r2 = (x - x0) ** 2 + (y - y0) ** 2
        surface = np.empty_like(r2, dtype=float)

        outside = r2 > R * R
        surface[outside] = z0 + R * R + z[outside]

        inside = ~outside
        if np.any(inside):
            surface[inside] = z0 - np.sqrt(R * R - r2[inside])

        gap = surface - z
        return gap

    def gap_n(self, pts, normals):
        nn = np.linalg.norm(normals, axis=1)
        if np.any(nn == 0.0):
            raise ValueError("Some normals have zero norm.")
        n_hat = normals / nn[:, None]

        c = self.center[None, :]
        R = self.radius

        r = pts - c
        s = np.linalg.norm(r, axis=1)

        q = np.empty_like(pts)
        mask = s > 0.0
        if np.any(mask):
            q[mask] = c + (R * (r[mask] / s[mask, None]))
        if np.any(~mask):
            q[~mask] = c + R * n_hat[~mask]

        g_vec = q - pts
        gap_n = np.einsum("ij,ij->i", g_vec, n_hat)
        return gap_n

    def new_gap_n(
        self,
        pts,
        normals,
        warn_dist=None,
        far_gap=None,
        warn_mode="clip",
    ):

        far_gap = self.radius if far_gap is None else far_gap

        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]

        normals = np.asarray(normals, dtype=float)
        if normals.ndim == 1:
            normals = normals[np.newaxis, :]

        if pts.shape != normals.shape:
            raise ValueError(
                f"normals must have the same shape as pts. Got {normals.shape} vs {pts.shape}."
            )

        nn = np.linalg.norm(normals, axis=1)
        if np.any(nn == 0.0):
            raise ValueError("Some normals have zero norm.")
        n_hat = normals / nn[:, None]

        c = self.center[None, :]
        R = self.radius

        r = pts - c
        s = np.linalg.norm(r, axis=1)

        # Closest point on sphere
        q = np.empty_like(pts)
        mask = s > 0.0
        if np.any(mask):
            q[mask] = c + (R * (r[mask] / s[mask, None]))
        if np.any(~mask):
            # Degenerate: p == c, choose direction from normal
            q[~mask] = c + R * n_hat[~mask]

        g_vec = q - pts
        gap_n = np.einsum("ij,ij->i", g_vec, n_hat)

        if warn_dist is not None:
            warn_dist = float(warn_dist)
            d = s - R  # signed radial distance to sphere surface
            far = d > warn_dist

            if np.any(far):
                if warn_mode == "warn":
                    # light warning; do not spam
                    print(
                        f"WARNING(Sphere.gap_n): {far.sum()} / {len(far)} points are farther than "
                        f"warn_dist={warn_dist:g} from the sphere surface (max d={d[far].max():.3e})."
                    )
                elif warn_mode == "clip":
                    gap_n[far] = float(far_gap)
                else:
                    raise ValueError("warn_mode must be 'clip' or 'warn'.")

        return gap_n
