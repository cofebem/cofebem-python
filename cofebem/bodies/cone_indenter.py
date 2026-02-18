import numpy as np


class Cone:
    def __init__(self, top_center, top_radius, height):
        self.top_center = np.array(top_center, dtype=float)
        self.radius = float(top_radius)
        self.height = float(height)

        if self.radius <= 0.0:
            raise ValueError("top_radius must be > 0")
        if self.height <= 0.0:
            raise ValueError("height must be > 0")

    @property
    def apex(self):
        c = self.top_center
        return np.array([c[0], c[1], c[2] - self.height], dtype=float)

    def gap(self, pts):
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        x0, y0, z_top = self.top_center
        R = self.radius
        H = self.height
        z_apex = z_top - H

        dx = x - x0
        dy = y - y0
        r = np.sqrt(dx * dx + dy * dy)

        surface = np.empty_like(r, dtype=float)

        outside = r > R
        surface[outside] = z_top + (R * R + H * H) + z[outside]

        inside = ~outside
        if np.any(inside):
            surface[inside] = z_apex + (H / R) * r[inside]

        gap = surface - z
        return gap

    def gap_n(self, pts, normals):

        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]

        normals = np.asarray(normals, dtype=float)
        if normals.ndim == 1:
            normals = normals[np.newaxis, :]

        if pts.shape != normals.shape:
            raise ValueError(
                f"normals must have same shape as pts. Got {normals.shape} vs {pts.shape}."
            )

        nn = np.linalg.norm(normals, axis=1)
        if np.any(nn == 0.0):
            raise ValueError("Some normals have zero norm.")
        n_hat = normals / nn[:, None]

        x0, y0, z_top = self.center
        R = self.radius
        H = self.height
        z_apex = z_top - H

        A = np.array([0.0, z_apex], dtype=float)
        B = np.array([R, z_top], dtype=float)
        AB = B - A
        AB2 = float(np.dot(AB, AB))

        dx = pts[:, 0] - x0
        dy = pts[:, 1] - y0
        r = np.sqrt(dx * dx + dy * dy)
        z = pts[:, 2]
        U = np.column_stack((r, z))  # (N,2)

        # project U onto segment AB
        t = ((U - A) @ AB) / AB2  # (N,)
        t = np.clip(t, 0.0, 1.0)
        Uproj = A[None, :] + t[:, None] * AB[None, :]  # (N,2)
        rq = Uproj[:, 0]
        zq = Uproj[:, 1]

        # map back to 3D: keep same azimuth direction in xy
        q = np.empty_like(pts)
        mask_r = r > 0.0
        if np.any(mask_r):
            scale = rq[mask_r] / r[mask_r]
            q[mask_r, 0] = x0 + scale * dx[mask_r]
            q[mask_r, 1] = y0 + scale * dy[mask_r]
        if np.any(~mask_r):
            # if point is exactly on axis, azimuth is undefined: use the normal's xy direction
            nxy = n_hat[~mask_r, :2]
            nxy_norm = np.linalg.norm(nxy, axis=1)
            # if normal has no xy component either, pick x direction
            use_default = nxy_norm == 0.0
            nxy_unit = np.empty_like(nxy)
            if np.any(~use_default):
                nxy_unit[~use_default] = (
                    nxy[~use_default] / nxy_norm[~use_default, None]
                )
            if np.any(use_default):
                nxy_unit[use_default] = np.array([1.0, 0.0])
            q[~mask_r, 0] = x0 + rq[~mask_r] * nxy_unit[:, 0]
            q[~mask_r, 1] = y0 + rq[~mask_r] * nxy_unit[:, 1]

        q[:, 2] = zq

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

        far_gap = self.height if far_gap is None else float(far_gap)

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

        # compute closest points q (same logic as gap_n, but we keep q for distance checks)
        x0, y0, z_top = self.center
        R = self.radius
        H = self.height
        z_apex = z_top - H

        A = np.array([0.0, z_apex], dtype=float)
        B = np.array([R, z_top], dtype=float)
        AB = B - A
        AB2 = float(np.dot(AB, AB))

        dx = pts[:, 0] - x0
        dy = pts[:, 1] - y0
        r = np.sqrt(dx * dx + dy * dy)
        z = pts[:, 2]
        U = np.column_stack((r, z))

        t = ((U - A) @ AB) / AB2
        t = np.clip(t, 0.0, 1.0)
        Uproj = A[None, :] + t[:, None] * AB[None, :]
        rq = Uproj[:, 0]
        zq = Uproj[:, 1]

        q = np.empty_like(pts)
        mask_r = r > 0.0
        if np.any(mask_r):
            scale = rq[mask_r] / r[mask_r]
            q[mask_r, 0] = x0 + scale * dx[mask_r]
            q[mask_r, 1] = y0 + scale * dy[mask_r]
        if np.any(~mask_r):
            nxy = n_hat[~mask_r, :2]
            nxy_norm = np.linalg.norm(nxy, axis=1)
            use_default = nxy_norm == 0.0
            nxy_unit = np.empty_like(nxy)
            if np.any(~use_default):
                nxy_unit[~use_default] = (
                    nxy[~use_default] / nxy_norm[~use_default, None]
                )
            if np.any(use_default):
                nxy_unit[use_default] = np.array([1.0, 0.0])
            q[~mask_r, 0] = x0 + rq[~mask_r] * nxy_unit[:, 0]
            q[~mask_r, 1] = y0 + rq[~mask_r] * nxy_unit[:, 1]
        q[:, 2] = zq

        g_vec = q - pts
        gap_n = np.einsum("ij,ij->i", g_vec, n_hat)

        if warn_dist is not None:
            warn_dist = float(warn_dist)
            d = np.linalg.norm(
                g_vec, axis=1
            )  # true 3D distance to closest point on cone surface
            far = d > warn_dist
            if np.any(far):
                if warn_mode == "warn":
                    print(
                        f"WARNING(Cone.gap_n): {far.sum()} / {len(far)} points are farther than "
                        f"warn_dist={warn_dist:g} from the cone surface (max d={d[far].max():.3e})."
                    )
                elif warn_mode == "clip":
                    gap_n[far] = far_gap
                else:
                    raise ValueError("warn_mode must be 'clip' or 'warn'.")

        return gap_n
