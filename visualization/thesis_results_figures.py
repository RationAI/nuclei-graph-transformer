"""Thesis results figures, redrawn from the saved outputs of a results run; nothing is recomputed.

`res_sources`, for shape, texture and position only, all with k-NN attention and rotated values, as
two separate images that fit side by side at the text width (use `subcaption` for the panel letters):
`res_sources_a`: AUPRC, one row per variant: the full-test-set point estimate (filled circle), its
    95 % bootstrap CI (horizontal bar) and the three seeds (small open circles). Each seed has its
    own height above the bar, so they cannot overlap and their AUPRC stays exact. The numbers are
    listed beside the plot.
`res_sources_b`: precision-recall curves: seed mean, seed min-max shaded.

Usage:
    uv run -m visualization.thesis_results_figures [--results-dir DIR] [--out-dir DIR]

Reads `configurations.csv`, `counts.csv` and `pr_curves.npz` of `--results-dir`
(postprocessing.results_sources).
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from numpy.typing import NDArray

from postprocessing.bootstrap_sources import THESIS_RC


RESULTS_DIR = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/q2_results"
)

TEX_PT_PER_IN = 72.27
TEXT_W_IN, FIG_H_IN = 455 / TEX_PT_PER_IN, 2.4
FIG_A_W_IN, FIG_B_W_IN = 0.4 * TEXT_W_IN, 0.6 * TEXT_W_IN
FONT_SIZE = 8
RC = {**THESIS_RC, "font.size": FONT_SIZE}

INK, GRID = "#1f1f1f", "#e6e6e6"
SHAPE, TEXTURE, POSITION = "#2C6E9B", "#B5651D", "#5E8C61"
MARKER_SIZE, SEED_SIZE = 3.4, 2.6
SEED_ROW_OFFSETS = (0.18, 0.28, 0.38)  # rows above the CI bar, one per seed

SOURCES = (
    ("shape_knn", "shape", SHAPE),
    ("texture_knn", "texture", TEXTURE),
    ("position_knn", "position only", POSITION),
)
AUPRC_RANGE = (0.80, 0.96)
AUPRC_TICKS = (0.80, 0.85, 0.90, 0.95)

# Axes in figure fractions, the same height in both images so they align side by side.
BOTTOM, HEIGHT = 0.17, 0.71
AX_A = (0.315, BOTTOM, 0.375, HEIGHT)
VALUES_X = 0.845  # centre of the numbers column of (a), figure fraction
AX_B = (0.135, BOTTOM, 0.845, HEIGHT)


def plot_row(ax: plt.Axes, y: float, row: pd.Series, color: str) -> None:
    """The point estimate with its horizontal CI, and the seeds above the bar."""
    ax.errorbar(
        row["point"],
        y,
        xerr=[[row["point"] - row["ci_lo"]], [row["ci_hi"] - row["point"]]],
        fmt="o",
        ms=MARKER_SIZE,
        color=color,
        mfc=color,
        mew=0.8,
        elinewidth=1.0,
        capsize=2.2,
        capthick=1.0,
        zorder=4,
    )
    seeds = [row[f"seed_{k + 1}"] for k in range(int(row["n_seeds"]))]
    ax.plot(
        seeds,
        y - np.array(SEED_ROW_OFFSETS[: len(seeds)]),  # the y axis runs top to bottom
        "o",
        ms=SEED_SIZE,
        mfc="none",
        mec=color,
        mew=0.6,
        ls="none",
        zorder=3,
    )


def clean_spines(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def auprc_figure(conf: pd.DataFrame) -> plt.Figure:
    rows = conf[conf["metric"] == "AUPRC"].set_index("configuration")
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(FIG_A_W_IN, FIG_H_IN))
        ax = fig.add_axes(AX_A)
        values = blended_transform_factory(fig.transFigure, ax.transData)
        for y, (name, _, color) in enumerate(SOURCES):
            row = rows.loc[name]
            plot_row(ax, y, row, color)
            ax.text(
                VALUES_X,
                y,
                f"{row['point']:.3f}\n[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
                transform=values,
                ha="center",
                va="center",
                color=INK,
                fontsize=FONT_SIZE - 1,
                linespacing=1.15,
            )
        ax.text(
            VALUES_X,
            BOTTOM + HEIGHT + 0.012,
            "AUPRC\n[95% CI]",
            transform=fig.transFigure,
            ha="center",
            va="bottom",
            color=INK,
            fontsize=FONT_SIZE - 1,
            linespacing=1.15,
        )
        ax.set_yticks(range(len(SOURCES)), [label for _, label, _ in SOURCES])
        ax.set_ylim(len(SOURCES) - 0.5, -0.6)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(*AUPRC_RANGE)
        ax.set_xticks(AUPRC_TICKS)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.set_xlabel("AUPRC")
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        clean_spines(ax)
    return fig


def pr_figure(
    curves: dict[str, NDArray], recall: NDArray, prevalence: float
) -> plt.Figure:
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(FIG_B_W_IN, FIG_H_IN))
        ax = fig.add_axes(AX_B)
        for name, label, color in SOURCES:
            c = curves[name]
            ax.fill_between(recall, c.min(0), c.max(0), color=color, alpha=0.25, lw=0)
            ax.plot(recall, c.mean(0), color=color, lw=1.1, label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(prevalence - 0.05, 1.005)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(
            loc="lower left", frameon=False, fontsize=FONT_SIZE - 1, handlelength=2.0
        )
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        clean_spines(ax)
    return fig


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Saves under the same rc as the figure was built: fonts are resolved when drawing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig.savefig(out_dir / f"{stem}.pdf")
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def save_sources_figures(
    conf: pd.DataFrame,
    curves: dict[str, NDArray],
    recall: NDArray,
    prevalence: float,
    out_dir: Path,
) -> None:
    save_figure(auprc_figure(conf), out_dir, "res_sources_a")
    save_figure(pr_figure(curves, recall, prevalence), out_dir, "res_sources_b")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"

    conf = pd.read_csv(args.results_dir / "configurations.csv")
    prevalence = pd.read_csv(args.results_dir / "counts.csv")["prevalence"].item()
    with np.load(args.results_dir / "pr_curves.npz") as z:
        curves = {k: z[k] for k in z.files if k != "recall"}
        recall = z["recall"]

    save_sources_figures(conf, curves, recall, prevalence, out_dir)
    print(f"{out_dir}/res_sources_a.pdf\n{out_dir}/res_sources_b.pdf")


if __name__ == "__main__":
    main()
