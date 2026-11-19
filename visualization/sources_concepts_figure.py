# ruff: noqa: RUF002  (mathematical symbols in text are intended)
"""fig:res-sources-concepts, redrawn from the saved outputs of postprocessing.sources_concepts.

One panel, laid out as fig:res-context-concepts. Test nuclei are grouped by the stable SAE concept
(A-H) dominating their contribution |φ_ij| ("other": an unstable concept dominates, or none is
active); the last column is all nuclei. Per group, four differences of mean log-loss
(seed-averaged probabilities, first − second): shape − texture without and with context, shape −
position only and texture − position only, each with the 95 % stratified slide-bootstrap CI. Below
0, the first source has the lower loss. Below the axis: nuclei per group and their carcinoma share.

Usage:
    uv run -m visualization.sources_concepts_figure [--results-dir DIR] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from visualization.context_concepts_figure import MUTED, count_label
from visualization.thesis_results_figures import (
    FONT_SIZE,
    GRID,
    INK,
    RC,
    SHAPE,
    TEXT_W_IN,
    TEXTURE,
    clean_spines,
    save_figure,
)


RESULTS_DIR = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/q2_sources_concepts"
)
FIGURE_STEM = "res_sources_concepts"
FIG_H_IN = 2.9
SHAPE_TEXTURE = "#6A4C93"
# (column, label, colour, marker, filled, x offset); mathtext minus (cmr10 has no Unicode glyph).
# Shape minus texture in one colour (open: self-only, filled: with context); the position-only
# comparisons in the colour of the source compared with position only (as in fig:res-sources).
SERIES = (
    (
        "shape_vs_texture_self",
        "shape $-$ texture, self-only",
        SHAPE_TEXTURE,
        "o",
        False,
        -0.27,
    ),
    (
        "shape_vs_texture_knn",
        "shape $-$ texture, with context",
        SHAPE_TEXTURE,
        "o",
        True,
        -0.09,
    ),
    ("shape_vs_position_knn", "shape $-$ position only", SHAPE, "s", True, 0.09),
    ("texture_vs_position_knn", "texture $-$ position only", TEXTURE, "D", True, 0.27),
)
AX = (0.1, 0.25, 0.88, 0.62)
ROW_Y = (-0.17, -0.28)  # rows under the axis (axes fraction): nuclei, carcinoma share


def sources_concepts_figure(table: pd.DataFrame) -> plt.Figure:
    rows = table.reset_index(drop=True)
    x = np.arange(len(rows), dtype=float)
    x[rows["group"] == "all"] += 0.3  # set apart from the concept groups
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(TEXT_W_IN, FIG_H_IN))
        ax = fig.add_axes(AX)
        ax.axhline(0, color=INK, lw=0.7, zorder=1)
        ax.axvline(x[-1] - 0.65, color=GRID, lw=0.8, zorder=0)
        for col, label, color, marker, filled, dx in SERIES:
            ax.errorbar(
                x + dx,
                rows[col],
                yerr=[
                    rows[col] - rows[f"{col}_ci_lo"],
                    rows[f"{col}_ci_hi"] - rows[col],
                ],
                fmt=marker,
                ms=3.2 if marker != "D" else 2.8,
                color=color,
                mfc=color if filled else "white",
                mec=color,
                mew=0.8,
                elinewidth=1.0,
                capsize=1.6,
                capthick=1.0,
                zorder=4,
                label=label,
            )
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
            fontsize=FONT_SIZE - 1,
            handlelength=1.2,
            ncols=4,
            borderaxespad=0.2,
            columnspacing=1.2,
        )
        ax.set_xticks(x, rows["group"].tolist())
        ax.tick_params(axis="x", length=0, pad=4)
        ax.set_xlim(-0.6, x[-1] + 0.6)
        ax.set_ylabel(r"$\Delta$ mean log-loss")
        ax.text(
            1.0,
            0.02,
            "below 0: the first source has the lower loss",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=FONT_SIZE - 1,
        )
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        clean_spines(ax)
        under = blended_transform_factory(ax.transData, ax.transAxes)
        for y, header, values in (
            (ROW_Y[0], "nuclei", [count_label(n) for n in rows["n_nuclei"]]),
            (
                ROW_Y[1],
                "carcinoma",
                [f"{s:.0%}".replace("%", " %") for s in rows["carcinoma_share"]],
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


def save_sources_concepts_figure(results_dir: Path, out_dirs: list[Path]) -> list[Path]:
    table = pd.read_csv(results_dir / "groups.csv")
    written = []
    for out_dir in out_dirs:
        save_figure(sources_concepts_figure(table), out_dir, FIGURE_STEM)
        written += [out_dir / f"{FIGURE_STEM}.pdf", out_dir / f"{FIGURE_STEM}.png"]
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    print(
        "\n".join(map(str, save_sources_concepts_figure(args.results_dir, [out_dir])))
    )


if __name__ == "__main__":
    main()
