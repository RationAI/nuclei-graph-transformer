# ruff: noqa: RUF002  (mathematical symbols in text are intended)
"""fig:res-context-concepts, redrawn from the saved outputs of postprocessing.context_concepts.

One panel: test nuclei grouped by the stable SAE concept (A-H) dominating their contribution
|φ_ij| ("other": an unstable concept dominates, or none is active). Per group, the difference of
mean log-loss k-NN − self-only and dense − k-NN (seed-averaged probabilities), with the 95 %
stratified slide-bootstrap CI; below 0, the larger context lowers the loss. Dashed lines: the
same differences over all nuclei. Below the axis: nuclei per group and their carcinoma share.

Usage:
    uv run -m visualization.context_concepts_figure [--results-dir DIR] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from visualization.thesis_results_figures import (
    FONT_SIZE,
    GRID,
    INK,
    RC,
    TEXT_W_IN,
    clean_spines,
    save_figure,
)


RESULTS_DIR = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/q3_context_concepts"
)
FIG_H_IN = 2.7
MUTED = "#6b6b6b"
# (column, label, colour, marker, x offset); mathtext minus/Delta (cmr10 has no Unicode glyphs); colours as fig:res-context (k-NN blue, dense ochre)
SERIES = (
    ("knn_vs_self", "k-NN $-$ self-only", "#2C6E9B", "o", -0.14),
    ("dense_vs_knn", "dense $-$ k-NN", "#C0882B", "s", 0.14),
)
AX = (0.1, 0.27, 0.88, 0.66)
ROW_Y = (-0.17, -0.27)  # rows under the axis (axes fraction): nuclei, carcinoma share


def count_label(n: int) -> str:
    return f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n / 1e3:.0f} k"


def context_concepts_figure(table: pd.DataFrame) -> plt.Figure:
    groups = table[table["group"] != "all"].reset_index(drop=True)
    overall = table[table["group"] == "all"].iloc[0]
    x = np.arange(len(groups))
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(TEXT_W_IN, FIG_H_IN))
        ax = fig.add_axes(AX)
        ax.axhline(0, color=INK, lw=0.7, zorder=1)
        for col, label, color, marker, dx in SERIES:
            ax.axhline(overall[col], color=color, lw=0.7, dashes=(4, 2), zorder=1)
            ax.errorbar(
                x + dx,
                groups[col],
                yerr=[
                    groups[col] - groups[f"{col}_ci_lo"],
                    groups[f"{col}_ci_hi"] - groups[col],
                ],
                fmt=marker,
                ms=3.4,
                color=color,
                mfc=color,
                mew=0.8,
                elinewidth=1.0,
                capsize=2.0,
                capthick=1.0,
                zorder=4,
                label=label,
            )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color=INK, lw=0.7, dashes=(4, 2)))
        labels.append("all nuclei")
        ax.legend(
            handles,
            labels,
            loc="lower right",
            frameon=False,
            fontsize=FONT_SIZE - 1,
            handlelength=2.0,
            ncols=3,
            borderaxespad=0.2,
        )
        ax.set_xticks(x, groups["group"].tolist())
        ax.tick_params(axis="x", length=0, pad=4)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylabel(r"$\Delta$ mean log-loss")
        ax.text(
            0.0,
            1.02,
            "below 0: the larger context lowers the loss",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=FONT_SIZE - 1,
        )
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        clean_spines(ax)
        under = blended_transform_factory(ax.transData, ax.transAxes)
        for y, header, values in (
            (ROW_Y[0], "nuclei", [count_label(n) for n in groups["n_nuclei"]]),
            (
                ROW_Y[1],
                "carcinoma",
                [f"{s:.0%}".replace("%", " %") for s in groups["carcinoma_share"]],
            ),
        ):
            ax.text(
                -0.75,
                y,
                header,
                transform=under,
                ha="right",
                va="center",
                color=INK,
                fontsize=FONT_SIZE - 1,
            )
            for xi, v in zip(x, values, strict=True):
                ax.text(
                    xi,
                    y,
                    v,
                    transform=under,
                    ha="center",
                    va="center",
                    color=INK,
                    fontsize=FONT_SIZE - 1,
                )
    return fig


def save_context_concepts_figure(results_dir: Path, out_dirs: list[Path]) -> list[Path]:
    table = pd.read_csv(results_dir / "groups.csv")
    written = []
    for out_dir in out_dirs:
        save_figure(context_concepts_figure(table), out_dir, "res_context_concepts")
        written += [
            out_dir / "res_context_concepts.pdf",
            out_dir / "res_context_concepts.png",
        ]
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    print(
        "\n".join(map(str, save_context_concepts_figure(args.results_dir, [out_dir])))
    )


if __name__ == "__main__":
    main()
