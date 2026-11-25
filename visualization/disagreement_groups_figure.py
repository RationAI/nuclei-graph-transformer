# ruff: noqa: RUF001, RUF002  (mathematical symbols in text are intended)
"""Group figures res_groups_sources and res_groups_context, from deltas.csv.

Saved by postprocessing.disagreement_groups. One panel per pair X vs Y, side by side; rows =
model-independent groups (Gleason pattern, carcinoma share of the slide, benign compartment,
distance to the nearest carcinoma nucleus, overall). Per row: Δ = metric_X − metric_Y in
percentage points (carcinoma: recall difference, filled circle; benign: specificity difference,
open square), 95 % stratified slide-bootstrap CI; zero line; positive = X better. Each model at
its own Youden threshold (in the panel subtitle). Nuclei per group left of the first panel.
Style as visualization.disagreement_net_figure.

Usage:
    uv run -m visualization.disagreement_groups_figure [--results-dir DIR] [--out-dir DIR]
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

from visualization.disagreement_net_figure import CLASS_STYLE, MUTED, count_label
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
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/disagreement_groups"
)
FIGURES = {
    "res_groups_sources": ["position_vs_shape", "shape_vs_texture"],
    "res_groups_context": ["self_vs_knn", "knn_vs_dense"],
}
FIG_H_IN = 3.55
AXES = ((0.255, 0.15, 0.33, 0.70), (0.655, 0.15, 0.33, 0.70))
LABEL_X, COUNT_X, SECTION_X = -0.17, -0.025, -0.70  # axes fraction (first panel)
HEADER_STEP, SECTION_GAP = 0.95, 0.45


def display(label: str) -> str:
    """Group label in glyphs the thesis text font (cmr10) has: comparisons, dash and µ as mathtext."""
    for a, b in (
        ("≤ ", "$\\leq$"),
        ("< ", "$<$"),
        ("> ", "$>$"),
        ("–", "$-$"),
        (" µm", " $\\mu$m"),
        (" %", "%"),
    ):
        label = label.replace(a, b)
    return label


def row_positions(part: pd.DataFrame) -> tuple[list[tuple[str, float]], list[float]]:
    """(section, header y) and the y of each row, top to bottom (y grows downwards)."""
    headers, ys, y, last = [], [], 0.0, None
    for section in part["section"]:
        if section != last:
            if last is not None:
                y += SECTION_GAP
            headers.append((section, y))
            y += HEADER_STEP
            last = section
        ys.append(y)
        y += 1.0
    return headers, ys


def panel(ax: plt.Axes, part: pd.DataFrame, first: bool) -> None:
    headers, ys = row_positions(part)
    ys = np.array(ys)
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    overall = part["section"].to_numpy() == "overall"
    if overall.any():
        y_sep = ys[overall].min() - HEADER_STEP - SECTION_GAP / 2
        ax.axhline(y_sep, color=GRID, lw=0.8, zorder=0)
    for cls, (color, marker, filled) in CLASS_STYLE.items():
        sel = (part["true_class"] == cls).to_numpy()
        r = part[sel]
        ax.errorbar(
            r["delta_pp"],
            ys[sel],
            xerr=[r["delta_pp"] - r["ci_lo"], r["ci_hi"] - r["delta_pp"]],
            fmt=marker,
            ms=3.0,
            color=color,
            mfc=color if filled else "white",
            mew=0.8,
            elinewidth=0.9,
            capsize=1.6,
            capthick=0.9,
            zorder=4,
        )
    ax.set_ylim(ys.max() + 0.7, -0.3)
    ax.set_yticks([])
    lo, hi = part["ci_lo"].min(), part["ci_hi"].max()
    pad = 0.08 * (hi - lo)
    ax.set_xlim(min(lo - pad, -1), max(hi + pad, 1))
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    clean_spines(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=FONT_SIZE - 1)
    x, yname = part["x"].iloc[0], part["y"].iloc[0]
    ax.set_xlabel(
        f"$\\Delta$ (percentage points); positive: {x} better", fontsize=FONT_SIZE - 1
    )
    ax.set_title(
        f"{part['question'].iloc[0]}: {x} vs {yname}",
        fontsize=FONT_SIZE,
        loc="left",
        pad=11,
    )
    ax.text(
        0,
        1.0,
        f"thresholds {part['threshold_x'].iloc[0]:.3f} vs "
        f"{part['threshold_y'].iloc[0]:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=FONT_SIZE - 2,
    )
    if not first:
        return
    side = blended_transform_factory(ax.transAxes, ax.transData)
    for section, y in headers:
        ax.text(
            SECTION_X,
            y,
            section,
            transform=side,
            ha="left",
            va="center",
            fontsize=FONT_SIZE - 1,
            color=INK,
            style="italic",
        )
    for y, r in zip(ys, part.itertuples(), strict=True):
        ax.text(
            LABEL_X,
            y,
            display(r.group),
            transform=side,
            ha="right",
            va="center",
            fontsize=FONT_SIZE - 1,
            color=INK,
        )
        ax.text(
            COUNT_X,
            y,
            count_label(r.n),
            transform=side,
            ha="right",
            va="center",
            fontsize=FONT_SIZE - 2.5,
            color=MUTED,
        )


def group_figure(table: pd.DataFrame, pairs: list[str]) -> plt.Figure:
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(TEXT_W_IN, FIG_H_IN))
        for i, (pair, rect) in enumerate(zip(pairs, AXES, strict=True)):
            panel(fig.add_axes(rect), table[table["pair"] == pair], first=i == 0)
        handles = [
            Line2D([], [], color=CLASS_STYLE["carcinoma"][0], marker="o", lw=0, ms=3.0),
            Line2D(
                [],
                [],
                color=CLASS_STYLE["benign"][0],
                marker="s",
                lw=0,
                ms=3.0,
                mfc="white",
                mew=0.8,
            ),
        ]
        fig.legend(
            handles,
            ["carcinoma: recall difference", "benign: specificity difference"],
            loc="upper left",
            bbox_to_anchor=(0.01, 1.0),
            frameon=False,
            ncols=2,
            fontsize=FONT_SIZE - 1,
            handletextpad=0.2,
        )
        fig.text(
            0.99,
            0.985,
            "own Youden thresholds (validation); nuclei per group at left",
            ha="right",
            va="top",
            color=MUTED,
            fontsize=FONT_SIZE - 1,
        )
    return fig


def save_group_figures(
    results_dir: Path, out_dirs: list[Path], figures: dict[str, list[str]] | None = None
) -> list[Path]:
    table = pd.read_csv(results_dir / "deltas.csv")
    written = []
    for out_dir in out_dirs:
        for stem, pairs in (figures or FIGURES).items():
            save_figure(group_figure(table, list(pairs)), out_dir, stem)
            written += [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    print("\n".join(map(str, save_group_figures(args.results_dir, [out_dir]))))


if __name__ == "__main__":
    main()
