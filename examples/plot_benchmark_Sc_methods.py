from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Categorical slots from the validated reference palette (dataviz skill).
# Color is assigned by entity and stays fixed across every figure:
# blue = flexibility, aqua = matrix-free, yellow/green/violet = contact size.
COLOR_FLEX = "#2a78d6"
COLOR_MATFREE = "#1baf7a"
CONTACT_COLORS = ["#eda100", "#008300", "#4a3aa7"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

METHOD_COLORS = {"flexibility": COLOR_FLEX, "matrix_free": COLOR_MATFREE}
METHOD_LABELS = {"flexibility": "Flexibility (dense Sc)", "matrix_free": "Matrix-free"}


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value in ("", None):
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _dofs_formatter(x, _pos):
    return f"{int(x):,}"


def plot_method_grid(
    rows: list[dict],
    value_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    value_fmt: str = "{:.3g}",
    yscale: str = "log",
) -> None:
    """Small-multiples grid: rows=contact fraction, cols=n_loads, lines=method."""
    contact_fractions = sorted({row["contact_side_fraction"] for row in rows})
    n_loads_values = sorted({int(row["n_loads"]) for row in rows})
    methods = ["flexibility", "matrix_free"]

    fig, axes = plt.subplots(
        len(contact_fractions),
        len(n_loads_values),
        figsize=(4.6 * len(n_loads_values), 3.1 * len(contact_fractions)),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    for i, cf in enumerate(contact_fractions):
        for j, n_loads in enumerate(n_loads_values):
            ax = axes[i][j]
            _style_axes(ax)
            cell_rows = [
                r
                for r in rows
                if r["contact_side_fraction"] == cf and int(r["n_loads"]) == n_loads
            ]
            lines = []
            for method in methods:
                mrows = sorted(
                    (r for r in cell_rows if r["method"] == method),
                    key=lambda r: r["n_dofs"],
                )
                if not mrows:
                    continue
                xs = [r["n_dofs"] for r in mrows]
                ys = [r[value_key] for r in mrows]
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=8,
                    linewidth=2,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
                lines.append((method, xs[-1], ys[-1]))

            # Stagger the end-of-line labels above/below so close-together
            # lines (e.g. near-identical memory use) don't smear together.
            lines.sort(key=lambda item: item[2])
            for rank, (method, x_last, y_last) in enumerate(lines):
                dy = 8 if rank == len(lines) - 1 else -8
                ax.annotate(
                    value_fmt.format(y_last),
                    (x_last, y_last),
                    textcoords="offset points",
                    xytext=(6, dy),
                    fontsize=8,
                    color=METHOD_COLORS[method],
                    va="bottom" if dy > 0 else "top",
                )

            ax.set_xscale("log")
            if yscale:
                ax.set_yscale(yscale)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_dofs_formatter))
            if i == 0:
                unit = "loading" if n_loads == 1 else "loadings"
                ax.set_title(f"{n_loads} {unit}", fontsize=10, color=INK_PRIMARY)
            if j == 0:
                ax.set_ylabel(f"contact side={cf:g}\n{ylabel}", fontsize=9)
            if i == len(contact_fractions) - 1:
                ax.set_xlabel("degrees of freedom", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=10,
        labelcolor=INK_PRIMARY,
    )
    fig.suptitle(title, fontsize=13, color=INK_PRIMARY, y=1.01)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_lu_factorization(rows: list[dict], out_path: Path) -> None:
    """LU factorization is a single solve of the elastic stiffness matrix,
    shared by both methods -- plot raw replicates plus the per-mesh median to
    show the two methods pay the same cost."""
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _style_axes(ax)

    for method in ("flexibility", "matrix_free"):
        mrows = [r for r in rows if r["method"] == method]
        ax.scatter(
            [r["n_dofs"] for r in mrows],
            [r["median_factorization_wall_s"] for r in mrows],
            s=24,
            color=METHOD_COLORS[method],
            alpha=0.35,
            linewidths=0,
            zorder=2,
        )
        by_dofs: dict[float, list[float]] = {}
        for r in mrows:
            by_dofs.setdefault(r["n_dofs"], []).append(r["median_factorization_wall_s"])
        xs = sorted(by_dofs)
        ys = [float(np.median(by_dofs[x])) for x in xs]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=8,
            linewidth=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_dofs_formatter))
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("LU factorization time, wall (s)")
    ax.set_title(
        "LU factorization cost is shared by both methods\n"
        "(dots: individual contact-fraction/load-count runs; line: median)",
        fontsize=11,
        color=INK_PRIMARY,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_break_even(rows: list[dict], out_path: Path) -> None:
    """Estimated number of load cases at which matrix-free overtakes
    flexibility, fit from the 1- and 10-loading measurements."""
    contact_fractions = sorted({row["contact_side_fraction"] for row in rows})
    mesh_sizes = sorted({row["n_dofs"] for row in rows})
    seen: dict[float, str] = {}
    for r in sorted(rows, key=lambda r: r["n_dofs"]):
        seen.setdefault(r["n_dofs"], f"{int(r['nx'])}×{int(r['ny'])}×{int(r['nz'])}")
    mesh_labels = [seen[d] for d in mesh_sizes]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _style_axes(ax)

    n_groups = len(contact_fractions)
    width = 0.8 / n_groups
    x = np.arange(len(mesh_sizes))

    for k, cf in enumerate(contact_fractions):
        values = []
        for dofs in mesh_sizes:
            match = [
                r
                for r in rows
                if r["n_dofs"] == dofs and r["contact_side_fraction"] == cf
            ]
            v = match[0]["estimated_break_even_n_solves"] if match else np.nan
            values.append(v if np.isfinite(v) else 0.0)
        offset = (k - (n_groups - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.9,
            color=CONTACT_COLORS[k % len(CONTACT_COLORS)],
            label=f"contact side={cf:g}",
        )
        for bar, v in zip(bars, values):
            if v > 0:
                ax.annotate(
                    f"{v:.0f}",
                    (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=8,
                    color=INK_SECONDARY,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(mesh_labels)
    ax.set_xlabel("mesh (nx×ny×nz)")
    ax.set_ylabel("estimated break-even number of load cases")
    ax.set_title(
        "Load count where matrix-free overtakes flexibility\n"
        "(affine fit from the 1- and 10-loading runs; below the bar, prefer flexibility)",
        fontsize=11,
        color=INK_PRIMARY,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "results" / "sc_benchmark_summary.csv",
    )
    parser.add_argument(
        "--break-even",
        type=Path,
        default=REPO_ROOT / "results" / "sc_benchmark_break_even.csv",
    )
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "plots")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = load_csv(args.summary)

    plot_method_grid(
        summary_rows,
        value_key="median_method_total_cpu_s",
        ylabel="total CPU time, setup+solve (s)",
        title="CPU time total: flexibility vs. matrix-free",
        out_path=args.outdir / "cpu_time_total.png",
    )

    plot_method_grid(
        summary_rows,
        value_key="median_rss_incremental_peak_mib",
        ylabel="peak memory over baseline (MiB)",
        title="Memory peak: flexibility vs. matrix-free",
        out_path=args.outdir / "memory_peak.png",
        yscale="linear",
        value_fmt="{:.1f}",
    )

    plot_method_grid(
        summary_rows,
        value_key="median_n_triangular_solves",
        ylabel="# triangular (LU back/forward) solves",
        title="Triangular-solve count: why the CPU-time gap exists",
        out_path=args.outdir / "triangular_solve_count.png",
        value_fmt="{:.0f}",
    )

    plot_lu_factorization(summary_rows, args.outdir / "lu_factorization_time.png")

    if args.break_even.exists():
        break_even_rows = load_csv(args.break_even)
        plot_break_even(break_even_rows, args.outdir / "break_even.png")
    else:
        print(f"Skipping break-even plot: {args.break_even} not found")

    print(f"Plots written to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
