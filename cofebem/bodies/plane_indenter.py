import numpy as np


class Plane:

    def __init__(self, point, normal=(0.0, 0.0, 1.0)):
        self.point = np.asarray(point, dtype=float)
        self.normal = np.asarray(normal, dtype=float)

        if self.point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {self.point.shape}.")
        if self.normal.shape != (3,):
            raise ValueError(f"normal must have shape (3,), got {self.normal.shape}.")

        nrm = np.linalg.norm(self.normal)
        if nrm == 0.0:
            raise ValueError("normal must be non-zero.")
        self.normal = self.normal / nrm

    def _as_points(self, pts):
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        if pts.shape[1] != 3:
            raise ValueError(f"pts must have shape (N, 3), got {pts.shape}.")
        return pts

    def gap(self, pts):

        pts = self._as_points(pts)
        signed_dist = (pts - self.point[None, :]) @ self.normal
        return signed_dist

    def gap_n(self, pts, normals):
        """
        Gap projected along supplied surface normals.
        """
        pts = self._as_points(pts)

        normals = np.asarray(normals, dtype=float)
        if normals.ndim == 1:
            normals = normals[np.newaxis, :]
        if normals.shape != pts.shape:
            raise ValueError(
                f"normals must have the same shape as pts. Got {normals.shape} vs {pts.shape}."
            )

        nn = np.linalg.norm(normals, axis=1)
        if np.any(nn == 0.0):
            raise ValueError("Some normals have zero norm.")
        n_hat = normals / nn[:, None]

        signed_dist = (pts - self.point[None, :]) @ self.normal
        g_vec = -signed_dist[:, None] * self.normal[None, :]
        return np.einsum("ij,ij->i", g_vec, n_hat)

    def new_gap_n(
        self,
        pts,
        normals,
        warn_dist=None,
        far_gap=None,
        warn_mode="clip",
    ):
        """
        Same as gap_n, with optional far-point handling for robustness.
        """
        pts = self._as_points(pts)

        normals = np.asarray(normals, dtype=float)
        if normals.ndim == 1:
            normals = normals[np.newaxis, :]
        if normals.shape != pts.shape:
            raise ValueError(
                f"normals must have the same shape as pts. Got {normals.shape} vs {pts.shape}."
            )

        nn = np.linalg.norm(normals, axis=1)
        if np.any(nn == 0.0):
            raise ValueError("Some normals have zero norm.")
        n_hat = normals / nn[:, None]

        signed_dist = (pts - self.point[None, :]) @ self.normal
        g_vec = -signed_dist[:, None] * self.normal[None, :]
        gap_n = np.einsum("ij,ij->i", g_vec, n_hat)

        if warn_dist is not None:
            warn_dist = float(warn_dist)
            d = np.abs(signed_dist)
            far = d > warn_dist
            if np.any(far):
                if warn_mode == "warn":
                    print(
                        f"WARNING(Plane.gap_n): {far.sum()} / {len(far)} points are farther than "
                        f"warn_dist={warn_dist:g} from the plane (max d={d[far].max():.3e})."
                    )
                elif warn_mode == "clip":
                    if far_gap is None:
                        far_gap = warn_dist
                    gap_n[far] = float(far_gap)
                else:
                    raise ValueError("warn_mode must be 'clip' or 'warn'.")

        return gap_n


class PlaneIndenter(Plane):
    pass
