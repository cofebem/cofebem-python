import numpy as np


def skew_from_vec(a: np.ndarray) -> np.ndarray:
    assert a.size == 3
    return np.array(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=a.dtype
    )


def exp_SO3_quat(z, normalize=True):
    z0, z_tilde = np.array_split(z, [1])
    Z = skew_from_vec(z_tilde)
    I = np.eye(3, dtype=z.dtype)

    if normalize:
        z_norm = z @ z
        return I + (2 / z_norm) * ((z0 * Z) + (Z @ Z))
    else:
        return I + 2 * ((z0 * Z) + (Z @ Z))


class RigidBody:
    def __init__(self, mesh, m, Ip, q0, name):
        self.mesh = mesh
        self.m = m
        self.Ip = Ip
        self.q0 = q0
        self.name = name
        self.nq = 7

    def R_IK(self):
        return exp_SO3_quat(self.q0[3:])

    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t, sol_i.q[self.qDOF])]
        vel = [self.v_P(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]
        omega = [
            self.A_IK(sol_i.t, sol_i.q[self.qDOF])
            @ self.K_Omega(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
        ]
        if sol_i.u_dot is not None:
            acc = [
                self.a_P(
                    sol_i.t,
                    sol_i.q[self.qDOF],
                    sol_i.u[self.uDOF],
                    sol_i.u_dot[self.uDOF],
                )
            ]
            psi = [
                self.A_IK(sol_i.t, sol_i.q[self.qDOF])
                @ self.K_Psi(
                    sol_i.t,
                    sol_i.q[self.qDOF],
                    sol_i.u[self.uDOF],
                    sol_i.u_dot[self.uDOF],
                )
            ]
        A_IK = np.vsplit(self.A_IK(sol_i.t, sol_i.q[self.qDOF]).T, 3)
        cells = [("vertex", [[0]])]
        if sol_i.u_dot is not None:
            cell_data = dict(
                v=[vel],
                Omega=[omega],
                a=[acc],
                psi=[psi],
                ex=[A_IK[0]],
                ey=[A_IK[1]],
                ez=[A_IK[2]],
            )
        else:
            cell_data = dict(
                v=[vel], Omega=[omega], ex=[A_IK[0]], ey=[A_IK[1]], ez=[A_IK[2]]
            )
        return points, cells, None, cell_data
