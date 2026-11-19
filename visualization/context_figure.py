# ruff: noqa: RUF003  (mathematical symbols in text are intended)
"""fig:res-context, redrawn from the saved outputs of postprocessing.results_context.

One figure at the text width, two panels, all test nuclei of the PANDA Radboud test set, shape
embedding without position encoding (self-only, k-NN with k = 24, dense):
(a) AUPRC per configuration: the seed-averaged point estimate (filled circle), its 95 % paired
    slide-level bootstrap CI (bar) and the three seeds (open circles, one row each above the bar);
    dashed lines for the thesis model (k-NN + RoVE, seed mean) and for the prevalence (the AUPRC of
    a random ranking). The numbers are listed beside the plot.
(b) Precision-recall curves: seed mean, seed min-max shaded; the thesis model's seed mean and
    the prevalence dashed.

Usage:
    uv run -m visualization.context_figure [--results-dir DIR] [--out-dir DIR]
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

from visualization.thesis_results_figures import (
    BOTTOM,
    FONT_SIZE,
    GRID,
    HEIGHT,
    INK,
    RC,
    TEXT_W_IN,
    clean_spines,
    plot_row,
    save_figure,
)


RESULTS_DIR = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/q3_context"
)
FIG_H_IN = 2.5
MUTED = "#8a8a8a"
# Checked for colour-vision deficiency (OKLab ΔE × 100 ≥ 14 for every pair, normal, protan,
# deutan, tritan); the thesis model is drawn in ink, not in a series colour.
CONFIGS = (
    ("self", "self-only", "#A8463F"),
    ("knn", "k-NN", "#2C6E9B"),
    ("dense", "dense", "#C0882B"),
)
THESIS_LABEL = "thesis model (k-NN + RoVE)"
AUPRC_RANGE = (0.35, 0.95)
AUPRC_TICKS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# axes in figure fractions
AX_A = (0.095, BOTTOM, 0.29, HEIGHT)
VALUES_X = 0.445  # centre of the numbers column of (a)
AX_B = (0.59, BOTTOM, 0.395, HEIGHT)


def panel_letter(fig: plt.Figure, ax: plt.Axes, letter: str, x: float) -> None:
    fig.text(
        x,
        ax.get_position().y1 + 0.035,
        f"({letter})",
        ha="left",
        va="bottom",
        color=INK,
        fontsize=FONT_SIZE + 1,
    )


def reference_vline(
    ax: plt.Axes, x: float, label: str, dashes: tuple, color: str
) -> None:
    """A dashed reference line, labelled along its left side from the bottom of the axes."""
    ax.axvline(x, color=color, lw=0.8, dashes=dashes, zorder=2)
    ax.text(
        x - 0.006,
        len(CONFIGS) - 0.55,
        label,
        rotation=90,
        ha="right",
        va="bottom",
        color=INK,
        fontsize=FONT_SIZE - 1.5,
    )


def context_figure(
    conf: pd.DataFrame,
    curves: dict[str, NDArray],
    recall: NDArray,
    prevalence: float,
) -> plt.Figure:
    rows = conf[(conf["subset"] == "all") & (conf["metric"] == "AUPRC")].set_index(
        "configuration"
    )
    thesis = rows.loc["thesis", "point"]
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(TEXT_W_IN, FIG_H_IN))

        # (a) AUPRC -------------------------------------------------------------
        ax = fig.add_axes(AX_A)
        values = blended_transform_factory(fig.transFigure, ax.transData)
        for y, (name, _, color) in enumerate(CONFIGS):
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
        reference_vline(ax, thesis, THESIS_LABEL, (4, 2), INK)
        reference_vline(ax, prevalence, "prevalence", (2, 2), MUTED)
        ax.set_yticks(range(len(CONFIGS)), [label for _, label, _ in CONFIGS])
        ax.set_ylim(len(CONFIGS) - 0.5, -0.65)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(*AUPRC_RANGE)
        ax.set_xticks(AUPRC_TICKS)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        ax.set_xlabel("AUPRC")
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        clean_spines(ax)
        panel_letter(fig, ax, "a", 0.01)

        # (b) precision-recall ----------------------------------------------------
        ax = fig.add_axes(AX_B)
        for name, label, color in CONFIGS:
            c = curves[name]
            ax.fill_between(recall, c.min(0), c.max(0), color=color, alpha=0.25, lw=0)
            ax.plot(recall, c.mean(0), color=color, lw=1.1, label=label)
        ax.plot(
            recall,
            curves["thesis"].mean(0),
            color=INK,
            lw=0.8,
            dashes=(4, 2),
            label="thesis model (k-NN + RoVE)",
        )
        ax.axhline(prevalence, color=MUTED, lw=0.8, dashes=(2, 2), label="prevalence")
        ax.set_xlim(0, 1)
        ax.set_ylim(prevalence - 0.05, 1.005)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 0.1),  # above the prevalence line
            frameon=False,
            fontsize=FONT_SIZE - 1,
            handlelength=2.0,
            borderaxespad=0.2,
        )
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        clean_spines(ax)
        panel_letter(fig, ax, "b", AX_B[0] - 0.075)
    return fig


def save_context_figure(results_dir: Path, out_dirs: list[Path]) -> list[Path]:
    conf = pd.read_csv(results_dir / "configurations.csv")
    counts = pd.read_csv(results_dir / "counts.csv").set_index("subset")
    with np.load(results_dir / "pr_curves.npz") as z:
        curves = {k[len("all/") :]: z[k] for k in z.files if k.startswith("all/")}
        recall = z["recall"]
    written = []
    for out_dir in out_dirs:
        fig = context_figure(conf, curves, recall, counts.loc["all", "prevalence"])
        save_figure(fig, out_dir, "res_context")
        written += [out_dir / "res_context.pdf", out_dir / "res_context.png"]
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    print("\n".join(map(str, save_context_figure(args.results_dir, [out_dir]))))


if __name__ == "__main__":
    main()
