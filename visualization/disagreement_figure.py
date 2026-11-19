
"""fig:res-disagreement, drawn from the saved outputs of postprocessing.disagreement_analysis.

One panel per pair of configurations (Q4: texture vs shape, shape vs position only; Q3: k-NN vs
self-only, dense vs k-NN). Per dominant concept (A-H, other): the enrichment of the concept in
the nuclei where the first configuration is correct and the second wrong (and the reverse),
relative to its share in the true class; carcinoma (filled circle) and benign (open square)
nuclei separately; 95 % stratified slide-bootstrap CI; log scale, line at 1 (no enrichment).
Points outside the axis are drawn as triangles at its edge. All disagreements (not only strong).

Usage:
    uv run -m visualization.disagreement_figure [--results-dir DIR] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

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
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/disagreement"
)
FIGURE_STEM = "res_disagreement"
FIG_H_IN = 4.6
MUTED = "#6b6b6b"
# direction colours as fig:res-context-concepts (blue, ochre); marker per class
DIRECTION_COLORS = {True: "#2C6E9B", False: "#C0882B"}
CLASS_MARKERS = {"carcinoma": ("o", True), "benign": ("s", False)}  # (marker, filled)
OFFSETS = {(True, "carcinoma"): -0.27, (True, "benign"): -0.09,
           (False, "carcinoma"): 0.09, (False, "benign"): 0.27}
Y_LIM = (1 / 256, 16)
Y_TICKS = (1 / 256, 1 / 64, 1 / 16, 1 / 4, 1, 4, 16)
PAIR_TITLES = {  # (title, first, second)
    "texture_vs_shape": ("Q4: texture vs shape", "texture", "shape"),
    "shape_vs_position": ("Q4: shape vs position only", "shape", "position only"),
    "knn_vs_self": ("Q3: k-NN vs self-only", "k-NN", "self-only"),
    "dense_vs_knn": ("Q3: dense vs k-NN", "dense", "k-NN"),
}


def tick_label(v: float) -> str:
    return f"{v:g}" if v >= 1 else f"1/{round(1 / v)}"


def panel(ax: plt.Axes, concepts: pd.DataFrame, pair: str) -> None:
    title, first, second = PAIR_TITLES[pair]
    part = concepts[(concepts["pair"] == pair) & (concepts["subset"] == "all")]
    groups = list(dict.fromkeys(part["group"]))
    x = np.arange(len(groups))
    lo_lim, hi_lim = Y_LIM
    ax.axhline(1, color=INK, lw=0.7, zorder=1)
    for (forward, cls), dx in OFFSETS.items():
        rows = part[(part["forward"] == forward) & (part["true_class"] == cls)]
        rows = rows.set_index("group").loc[groups]
        color = DIRECTION_COLORS[forward]
        marker, filled = CLASS_MARKERS[cls]
        v = rows["enrichment"].to_numpy(float)
        lo = np.clip(rows["ci_lo"].to_numpy(float), lo_lim, hi_lim)
        hi = np.clip(rows["ci_hi"].to_numpy(float), lo_lim, hi_lim)
        inside = np.isfinite(v) & (v >= lo_lim) & (v <= hi_lim)
        ax.errorbar(
            (x + dx)[inside],
            v[inside],
            yerr=[v[inside] - lo[inside], hi[inside] - v[inside]],
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
        for xi, vi in zip(x + dx, v, strict=True):
            if np.isfinite(vi) and not lo_lim <= vi <= hi_lim:
                edge, tri = (hi_lim, "^") if vi > hi_lim else (lo_lim, "v")
                ax.plot(xi, edge, tri, ms=3.4, color=color,
                        mfc=color if filled else "white", mew=0.8, zorder=5, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(*Y_LIM)
    ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([tick_label(t) for t in Y_TICKS])
    ax.set_xticks(x, groups)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    clean_spines(ax)
    ax.set_title(title, fontsize=FONT_SIZE, loc="left", pad=12)
    handles = [Line2D([], [], color=DIRECTION_COLORS[f], marker="o", lw=0, ms=3.0)
               for f in (True, False)]
    ax.legend(handles, [rf"{first}$\,{{>}}\,${second}", rf"{second}$\,{{>}}\,${first}"], loc="lower left",
              bbox_to_anchor=(0.0, 0.99), frameon=False, ncols=2, fontsize=FONT_SIZE - 1,
              handletextpad=0.2, columnspacing=1.0, borderaxespad=0.0, borderpad=0.0)


def disagreement_figure(concepts: pd.DataFrame) -> plt.Figure:
    pairs = list(dict.fromkeys(concepts["pair"]))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(TEXT_W_IN, FIG_H_IN), sharey=True)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.86, bottom=0.07, hspace=0.62,
                            wspace=0.08)
        for ax, pair in zip(axes.ravel(), pairs, strict=True):
            panel(ax, concepts, pair)
        for ax in axes[:, 0]:
            ax.set_ylabel("enrichment (set / class)")
        handles = [
            Line2D([], [], color=INK, marker="o", lw=0, ms=3.0, mfc=INK),
            Line2D([], [], color=INK, marker="s", lw=0, ms=3.0, mfc="white", mew=0.8),
        ]
        fig.legend(handles, ["carcinoma nuclei", "benign nuclei"], loc="upper left",
                   bbox_to_anchor=(0.1, 1.0), frameon=False, ncols=2,
                   fontsize=FONT_SIZE - 1, handletextpad=0.2)
        fig.text(0.99, 0.985, r"X$\,{>}\,$Y: X correct, Y wrong (threshold 0.40); 1 = same share as in the class",
                 ha="right", va="top", color=MUTED, fontsize=FONT_SIZE - 1)
    return fig


def save_disagreement_figure(results_dir: Path, out_dirs: list[Path]) -> list[Path]:
    concepts = pd.read_csv(results_dir / "concepts.csv")
    written = []
    for out_dir in out_dirs:
        save_figure(disagreement_figure(concepts), out_dir, FIGURE_STEM)
        written += [out_dir / f"{FIGURE_STEM}.pdf", out_dir / f"{FIGURE_STEM}.png"]
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    print("\n".join(map(str, save_disagreement_figure(args.results_dir, [out_dir]))))


if __name__ == "__main__":
    main()
