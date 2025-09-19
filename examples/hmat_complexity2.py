# benchmark_hmatrix_full.py
import time
from typing import Optional, Tuple, List, Callable

import numpy as np
import meshio
import matplotlib.pyplot as plt

from cofebem.hmatrices.hmatrix import HMatrix


def MatVec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Handmade dense matvec with Python loops: O(mn) and slow on purpose."""
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
    # One warm-up (esp. for JIT / cache warm)
    func(*args, **kwargs)
    acc = 0.0
    for _ in range(repeats):
        acc += time_once(func, *args, **kwargs)
    return acc / max(repeats, 1)


def metrics_from_hm(hm: HMatrix) -> dict:
    """Collect block-structure metrics (no timing)."""
    blocks = hm.block_tree.blocks
    nB = len(blocks)
    nLR = 0
    ranks = []
    lr_ops = 0  # proxy sum_k (m_k + n_k) * r_k
    dn_ops = 0  # proxy sum_k m_k * n_k

    for bl in blocks:
        m, n = bl.shape
        if bl.kind == "lr":
            nLR += 1
            r = (
                int(bl.rank)
                if bl.rank is not None
                else (bl.U.shape[1] if bl.U is not None else 0)
            )
            ranks.append(r)
            lr_ops += (m + n) * r
        else:
            dn_ops += m * n

    lr_frac = nLR / nB if nB else 0.0
    rank_median = float(np.median(ranks)) if ranks else 0.0
    rank_max = float(np.max(ranks)) if ranks else 0.0

    return dict(
        blocks=nB,
        lr_frac=lr_frac,
        rank_median=rank_median,
        rank_max=rank_max,
        lr_ops=float(lr_ops),
        dn_ops=float(dn_ops),
    )


def benchmark_grid(
    pts: np.ndarray,
    A: np.ndarray,
    leaf_grid: List[int],
    eta_grid: List[float],
    tol: float = 1e-6,
    split: str = "pca",
    repeats: int = 3,
) -> Tuple[
    np.ndarray,  # times_H
    np.ndarray,  # times_overhead
    np.ndarray,  # times_py (handmade)
    np.ndarray,  # times_blas (A@v)
    np.ndarray,  # rel_errs
    dict,  # metrics heatmaps
]:
    rng = np.random.default_rng(42)
    n = len(pts)
    v = rng.standard_normal(n)

    t_py = time_average(MatVec, repeats, A, v)
    t_blas = time_average(lambda M, x: M @ x, repeats, A, v)

    times_H = np.zeros((len(eta_grid), len(leaf_grid)), dtype=float)
    times_overhead = np.zeros_like(times_H)
    times_py = np.full_like(times_H, t_py, dtype=float)
    times_blas = np.full_like(times_H, t_blas, dtype=float)
    rel_errs = np.zeros_like(times_H)

    blocks_grid = np.zeros_like(times_H)
    lr_frac_grid = np.zeros_like(times_H)
    rank_median_grid = np.zeros_like(times_H)
    rank_max_grid = np.zeros_like(times_H)
    lr_ops_grid = np.zeros_like(times_H)
    dn_ops_grid = np.zeros_like(times_H)

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

            tH = time_average(lambda vec: hm @ vec, repeats, v)
            tOver = time_average(lambda vec: hm.matvec_overhead(vec), repeats, v)

            yH = hm @ v
            yA = A @ v
            rel = np.linalg.norm(yH - yA) / max(1e-16, np.linalg.norm(yA))

            times_H[ie, il] = tH
            times_overhead[ie, il] = tOver
            rel_errs[ie, il] = rel

            m = metrics_from_hm(hm)
            blocks_grid[ie, il] = m["blocks"]
            lr_frac_grid[ie, il] = m["lr_frac"]
            rank_median_grid[ie, il] = m["rank_median"]
            rank_max_grid[ie, il] = m["rank_max"]
            lr_ops_grid[ie, il] = m["lr_ops"]
            dn_ops_grid[ie, il] = m["dn_ops"]

    metrics = dict(
        blocks=blocks_grid,
        lr_frac=lr_frac_grid,
        rank_median=rank_median_grid,
        rank_max=rank_max_grid,
        lr_ops=lr_ops_grid,
        dn_ops=dn_ops_grid,
    )

    return times_H, times_overhead, times_py, times_blas, rel_errs, metrics


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


def compare_largest_lr_block(hm, repeats: int = 10, seed: int = 0):

    lr_blocks = [bl for bl in hm.block_tree.blocks if bl.kind == "lr"]
    if not lr_blocks:
        print("No low-rank blocks found.")
        return None

    b = max(lr_blocks, key=lambda bl: bl.shape[0] * bl.shape[1])
    i, j = b.row.idx, b.col.idx
    U, V = b.U, b.V
    m, n = len(i), len(j)
    r = U.shape[1]

    A = hm.block_tree.A
    B = A[np.ix_(i, j)]

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)

    y_dense = B @ x
    y_lr = U @ (V.T @ x)

    abs_err = np.linalg.norm(y_dense - y_lr)
    rel_err = abs_err / max(1e-16, np.linalg.norm(y_dense))

    def tavg(func, *args, repeats=repeats):
        func(*args)  # warm-up
        acc = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            func(*args)
            acc += time.perf_counter() - t0
        return acc / repeats

    def dense_mv(B, x):
        return B @ x

    def lr_mv(U, V, x):
        return U @ (V.T @ x)

    t_dense = tavg(dense_mv, B, x)
    t_lr = tavg(lr_mv, U, V, x)

    # 7) report
    print("Largest LR block:")
    print(f"  shape: ({m}, {n}), rank: {r}, area: {m*n}")
    print(f"  abs error: {abs_err:.3e}, rel error: {rel_err:.3e}")
    print(f"  time dense (B@x):     {t_dense*1e3:.3f} ms")
    print(f"  time low-rank (U V^T):{t_lr*1e3:.3f} ms")
    if t_lr > 0:
        print(f"  speedup (dense / LR): {t_dense / t_lr:.3f}×")

    return {
        "shape": (m, n),
        "rank": r,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "t_dense": t_dense,
        "t_lr": t_lr,
        "speedup_dense_over_lr": (t_dense / t_lr) if t_lr > 0 else np.inf,
    }


if __name__ == "__main__":
    mesh = meshio.read("hollow_cylinder.xdmf")
    pts = mesh.points
    # cells = mesh.cells  # unused here

    X = pts[:, None, :]
    Y = pts[None, :, :]
    A_full = 1.0 / (np.linalg.norm(X - Y, axis=2) + 1e-8)

    leaf_grid = [8, 16, 32, 64]
    eta_grid = [0.5, 0.7, 1.0, 1.5]
    repeats = 5
    tol = 1e-6
    split = "pca"  # try "kd" as well

    hm = HMatrix(
        pts, A_full, leaf_size=32, eta=1.0, tol=1e-6, split="pca", lr_approx="aca_full"
    )
    result = compare_largest_lr_block(hm, repeats=20)

    print("\nRunning grid benchmark...")
    (times_H, times_overhead, times_py, times_blas, rel_errs, metrics) = benchmark_grid(
        pts, A_full, leaf_grid, eta_grid, tol=tol, split=split, repeats=repeats
    )

    overhead_ratio = times_overhead / np.maximum(times_H, 1e-16)
    compute_only = np.maximum(times_H - times_overhead, 0.0)

    speedup_handmade = times_py / np.maximum(times_H, 1e-16)
    speedup_blas = times_blas / np.maximum(times_H, 1e-16)

    plot_heatmap(
        times_H,
        leaf_grid,
        eta_grid,
        "H-matrix matvec time (s)",
        "seconds",
        "hm_time_grid.png",
    )
    plot_heatmap(
        times_overhead,
        leaf_grid,
        eta_grid,
        "H-matrix overhead-only time (s)",
        "seconds",
        "hm_overhead_time_grid.png",
    )
    plot_heatmap(
        overhead_ratio,
        leaf_grid,
        eta_grid,
        "Overhead fraction: overhead/total",
        "fraction",
        "hm_overhead_ratio_grid.png",
    )
    plot_heatmap(
        compute_only,
        leaf_grid,
        eta_grid,
        "H-matrix compute-only time (s)",
        "seconds",
        "hm_compute_only_grid.png",
    )

    plot_heatmap(
        times_py,
        leaf_grid,
        eta_grid,
        "Handmade (Python-loop) time (s)",
        "seconds",
        "handmade_time_grid.png",
    )
    plot_heatmap(
        times_blas,
        leaf_grid,
        eta_grid,
        "BLAS (A @ v) time (s)",
        "seconds",
        "blas_time_grid.png",
    )

    plot_heatmap(
        speedup_handmade,
        leaf_grid,
        eta_grid,
        "Speedup: handmade / H-matvec",
        "× faster",
        "speedup_handmade_grid.png",
    )
    plot_heatmap(
        speedup_blas,
        leaf_grid,
        eta_grid,
        "Speedup: BLAS / H-matvec",
        "× faster",
        "speedup_blas_grid.png",
    )

    plot_heatmap(
        rel_errs,
        leaf_grid,
        eta_grid,
        "Relative error ‖Hv−Av‖/‖Av‖",
        "error",
        "rel_error_grid.png",
    )

    plot_heatmap(
        metrics["blocks"],
        leaf_grid,
        eta_grid,
        "Number of blocks",
        "count",
        "blocks_grid.png",
    )
    plot_heatmap(
        metrics["lr_frac"],
        leaf_grid,
        eta_grid,
        "Fraction of LR blocks",
        "ratio",
        "lr_frac_grid.png",
    )
    plot_heatmap(
        metrics["rank_median"],
        leaf_grid,
        eta_grid,
        "Median LR rank",
        "rank",
        "rank_median_grid.png",
    )
    plot_heatmap(
        metrics["rank_max"],
        leaf_grid,
        eta_grid,
        "Max LR rank",
        "rank",
        "rank_max_grid.png",
    )
    plot_heatmap(
        metrics["lr_ops"],
        leaf_grid,
        eta_grid,
        "LR ops proxy Σ(m+n)r",
        "units",
        "lr_ops_grid.png",
    )
    plot_heatmap(
        metrics["dn_ops"],
        leaf_grid,
        eta_grid,
        "Dense ops proxy Σmn",
        "units",
        "dn_ops_grid.png",
    )

    print(
        "\nDone. Saved: "
        "hm_time_grid.png, hm_overhead_time_grid.png, hm_overhead_ratio_grid.png, hm_compute_only_grid.png, "
        "handmade_time_grid.png, blas_time_grid.png, speedup_handmade_grid.png, speedup_blas_grid.png, "
        "rel_error_grid.png, blocks_grid.png, lr_frac_grid.png, rank_median_grid.png, rank_max_grid.png, "
        "lr_ops_grid.png, dn_ops_grid.png"
    )
