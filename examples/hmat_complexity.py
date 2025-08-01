import time
from typing import Optional, Tuple, List, Callable

import numpy as np
import meshio
import matplotlib.pyplot as plt

from cofebem.hmatrices.hmatrix import HMatrix


def MatVec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    m, n = A.shape

    if x.ndim == 2:
        if x.shape[1] != 1:
            raise ValueError("x must be (n,) or (n,1).")
        x_flat = x[:, 0]
    elif x.ndim == 1:
        x_flat = x
    else:
        raise ValueError("x must be (n,) or (n,1).")

    if x_flat.shape[0] != n:
        raise ValueError("Dimension mismatch: A is (m, n) but x length is not n.")

    b = np.empty(m, dtype=A.dtype)

    for i in range(m):
        acc = 0.0
        for j in range(n):
            acc += A[i, j] * x_flat[j]
        b[i] = acc
    return b


def time_once(func: Callable, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    func(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0


def time_average(func: Callable, repeats: int, *args, **kwargs) -> float:
    # One warm-up (especially for Numba JIT)
    func(*args, **kwargs)
    acc = 0.0
    for _ in range(repeats):
        acc += time_once(func, *args, **kwargs)
    return acc / repeats


def benchmark_grid(
    pts: np.ndarray,
    A: np.ndarray,
    leaf_grid: List[int],
    eta_grid: List[float],
    tol: float = 1e-6,
    split: str = "pca",
    repeats: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    n = len(pts)
    v = np.random.randn(n)

    t_py = time_average(MatVec, repeats, A, v)

    times_H = np.zeros((len(eta_grid), len(leaf_grid)), dtype=float)
    times_py = np.zeros_like(times_H)
    rel_errs = np.zeros_like(times_H)

    for ie, eta in enumerate(eta_grid):
        for il, leaf in enumerate(leaf_grid):
            hm = HMatrix(
                pts,
                A,
                leaf_size=leaf,
                eta=eta,
                tol=tol,
                split=split,
                lr_approx="aca_full",
            )
            # average apply time
            t_h = time_average(lambda vec: hm @ vec, repeats, v)
            yH = hm @ v
            yA = A @ v
            rel = np.linalg.norm(yH - yA) / max(1e-16, np.linalg.norm(yA))

            times_H[ie, il] = t_h
            times_py[ie, il] = t_py  # same value across the grid
            rel_errs[ie, il] = rel

    return times_H, times_py, rel_errs


def plot_heatmap(
    Z: np.ndarray,
    x_ticks: List,
    y_ticks: List,
    title: str,
    cbar_label: str,
    fname: Optional[str] = None,
):
    plt.figure(figsize=(7.5, 5.5))
    plt.imshow(Z, origin="lower", aspect="auto")
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.yticks(np.arange(len(y_ticks)), y_ticks)
    plt.xlabel("leaf_size")
    plt.ylabel("eta")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()


if __name__ == "__main__":

    mesh = meshio.read("hollow_cylinder.xdmf")
    pts = mesh.points
    cells = mesh.cells

    X = pts[:, None, :]
    Y = pts[None, :, :]
    A_full = 1.0 / (np.linalg.norm(X - Y, axis=2) + 1e-8)

    leaf_grid = [8, 16, 32, 64]
    eta_grid = [0.5, 0.7, 1.0, 1.5]
    repeats = 5

    print("\nRunning grid benchmark...")
    times_H, times_py, rel_errs = benchmark_grid(
        pts, A_full, leaf_grid, eta_grid, tol=1e-6, split="pca", repeats=repeats
    )

    speedup = times_py / np.maximum(times_H, 1e-16)

    plot_heatmap(
        times_H,
        leaf_grid,
        eta_grid,
        title="H-matrix matvec time (s)",
        cbar_label="seconds",
        fname="hm_time_grid.png",
    )

    plot_heatmap(
        times_py,
        leaf_grid,
        eta_grid,
        title="Handmade matvec time (s)",
        cbar_label="seconds",
        fname="handmade_time_grid.png",
    )

    plot_heatmap(
        speedup,
        leaf_grid,
        eta_grid,
        title="Speedup: handmade / H-matvec",
        cbar_label="× faster",
        fname="speedup_grid.png",
    )

    plot_heatmap(
        rel_errs,
        leaf_grid,
        eta_grid,
        title="Relative error ‖Hv−Av‖/‖Av‖",
        cbar_label="error",
        fname="rel_error_grid.png",
    )

    print(
        "\nDone. Saved: hm_time_grid.png, handmade_time_grid.png, speedup_grid.png, rel_error_grid.png"
    )
