from lcp_solvers import lemke, ccg, psor, nnls


class Contact:
    def __init__(
        self,
        body1,
        body2,
        g0,
        mu_f,
        lcp_solver="CCG",
    ):
        self.body1 = body1
        self.body2 = body2

        self_Gamma_c1 = body1.Gamma_c
        self_Gamma_c2 = body2.Gamma_c

        self.g0 = g0
        self.mu_f = mu_f

        self.lcp_solver = lcp_solver

    def g_N(self):
        pass
