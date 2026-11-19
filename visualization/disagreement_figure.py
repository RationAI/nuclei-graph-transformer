# ruff: noqa: RUF002  (mathematical symbols in text are intended)
"""fig:res-disagreement-context and fig:res-disagreement-sources (run cdaf1247).

Drawn from the saved outputs of postprocessing.disagreement_analysis.

Two figures, one row per pair of configurations: context (Q3: k-NN vs self-only, dense vs k-NN)
and sources (Q4: texture vs shape, shape vs position only). All disagreements (the strong ones,
|p_X − p_Y| ≥ 0.5, are in the CSVs only).

(a) Per dominant concept (A-H, other): the enrichment of the concept in the nuclei where the
first configuration is correct and the second wrong (and the reverse), relative to its share in
the true class; carcinoma (filled circle) and benign (open square) nuclei separately; 95 %
stratified slide-bootstrap CI; log scale, line at 1. Points outside the axis are drawn as
triangles at its edge. The legend gives each direction's set size per class (share of the class).
(b) The epithelial share of the benign nuclei in each direction's set, 95 % CI; dashed line =
the share in all benign nuclei.

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
FIGURES = {  # stem -> pairs (one row each)
    "res_disagreement_context": ("knn_vs_self", "dense_vs_knn"),
    "res_disagreement_sources": ("texture_vs_shape", "shape_vs_position"),
}
FIG_H_IN = 5.0
MUTED = "#6b6b6b"
# direction colours as fig:res-context-concepts (blue, ochre); marker per class
DIRECTION_COLORS = {True: "#2C6E9B", False: "#C0882B"}
CLASS_MARKERS = {"carcinoma": ("o", True), "benign": ("s", False)}  # (marker, filled)
OFFSETS = {(True, "carcinoma"): -0.27, (True, "benign"): -0.09,
           (False, "carcinoma"): 0.09, (False, "benign"): 0.27}
Y_LIM = (1 / 256, 16)
Y_TICKS = (1 / 256, 1 / 64, 1 / 16, 1 / 4, 1, 4, 16)
EPI_LIM = (0, 70)
EPI_TICKS = (0, 20, 40, 60)
PAIR_TITLES = {  # (title, first, second)
    "texture_vs_shape": ("Q4: texture vs shape", "texture", "shape"),
    "shape_vs_position": ("Q4: shape vs position only", "shape", "position only"),
    "knn_vs_self": ("Q3: k-NN vs self-only", "k-NN", "self-only"),
    "dense_vs_knn": ("Q3: dense vs k-NN", "dense", "k-NN"),
}
# figure fractions
LEFT, ENRICH_W, GAP, EPI_W = 0.1, 0.62, 0.085, 0.19
ROW_BOTTOMS, AX_H = (0.56, 0.08), 0.25
GT = r"$\,{>}\,$"


def tick_label(v: float) -> str:
    return f"{v:g}" if v >= 1 else f"1/{round(1 / v)}"


def pct(v: float) -> str:
    return f"{100 * v:.1f} %"


def enrichment_panel(ax: plt.Axes, concepts: pd.DataFrame, sets: pd.DataFrame,
                     pair: str) -> None:
    _, first, second = PAIR_TITLES[pair]
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
            (x + dx)[inside], v[inside],
            yerr=[v[inside] - lo[inside], hi[inside] - v[inside]],
            fmt=marker, ms=3.0, color=color, mfc=color if filled else "white", mew=0.8,
            elinewidth=0.9, capsize=1.6, capthick=0.9, zorder=4,
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
    ax.set_ylabel("enrichment (set / class)")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    clean_spines(ax)

    # legend = the two directions with their set sizes per class (share of the class)
    s = sets[(sets["pair"] == pair) & (sets["subset"] == "all")]
    labels, handles = [], []
    for forward, (a, b) in ((True, (first, second)), (False, (second, first))):
        cells = []
        for cls in ("carcinoma", "benign"):
            r = s[(s["forward"] == forward) & (s["true_class"] == cls)].iloc[0]
            cells.append(f"{cls} {r.n:,} ({pct(r.share_of_class)})")
        labels.append(f"{a}{GT}{b}:  " + ",  ".join(cells))
        handles.append(Line2D([], [], color=DIRECTION_COLORS[forward], marker="o", lw=0,
                              ms=3.0))
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              ncols=1, fontsize=FONT_SIZE - 1, handletextpad=0.3, borderaxespad=0.15,
              borderpad=0.0, labelspacing=0.25)


def epithelium_panel(ax: plt.Axes, epi: pd.DataFrame, pair: str) -> None:
    _, first, second = PAIR_TITLES[pair]
    part = epi[(epi["pair"] == pair) & (epi["subset"] == "all") & (epi["true_class"] == "benign")]
    class_share = part["epithelial_share_class"].iloc[0]
    ax.axhline(100 * class_share, color=INK, lw=0.7, dashes=(4, 2), zorder=1)
    ticks = []
    for xi, (forward, (a, b)) in enumerate(((True, (first, second)), (False, (second, first)))):
        r = part[part["forward"] == forward].iloc[0]
        v, lo, hi = 100 * r.epithelial_share_set, 100 * r.set_ci_lo, 100 * r.set_ci_hi
        color = DIRECTION_COLORS[forward]
        ax.errorbar(xi, v, yerr=[[v - lo], [hi - v]], fmt="s", ms=3.0, color=color,
                    mfc="white", mew=0.8, elinewidth=0.9, capsize=1.6, capthick=0.9, zorder=4)
        ticks.append(f"{a}{GT}\n{b}")
    ax.set_xticks([0, 1], ticks, fontsize=FONT_SIZE - 2)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(*EPI_LIM)
    ax.set_yticks(EPI_TICKS)
    ax.set_ylabel("epithelial share (%)")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    clean_spines(ax)
    ax.legend([Line2D([], [], color=INK, lw=0.7, dashes=(4, 2))],
              [f"all benign {pct(class_share)}"], loc="upper right", frameon=False,
              fontsize=FONT_SIZE - 2, handlelength=1.8, borderaxespad=0.1)


def disagreement_figure(concepts: pd.DataFrame, sets: pd.DataFrame, epi: pd.DataFrame,
                        pairs: tuple[str, ...]) -> plt.Figure:
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(TEXT_W_IN, FIG_H_IN))
        for pair, bottom in zip(pairs, ROW_BOTTOMS, strict=True):
            ax_e = fig.add_axes((LEFT, bottom, ENRICH_W, AX_H))
            ax_p = fig.add_axes((LEFT + ENRICH_W + GAP, bottom, EPI_W, AX_H))
            enrichment_panel(ax_e, concepts, sets, pair)
            epithelium_panel(ax_p, epi, pair)
            top = bottom + AX_H + 0.085
            fig.text(LEFT - 0.07, top, PAIR_TITLES[pair][0], ha="left", va="bottom",
                     fontsize=FONT_SIZE)
            for x0, label in ((LEFT, "(a)"), (LEFT + ENRICH_W + GAP, "(b)")):
                fig.text(x0 - 0.07, bottom + AX_H + 0.035, label, ha="left", va="bottom",
                         fontsize=FONT_SIZE)
        handles = [
            Line2D([], [], color=INK, marker="o", lw=0, ms=3.0, mfc=INK),
            Line2D([], [], color=INK, marker="s", lw=0, ms=3.0, mfc="white", mew=0.8),
        ]
        fig.legend(handles, ["carcinoma nuclei", "benign nuclei"], loc="upper left",
                   bbox_to_anchor=(LEFT - 0.01, 1.0), frameon=False, ncols=2,
                   fontsize=FONT_SIZE - 1, handletextpad=0.2)
        fig.text(0.99, 0.985,
                 rf"X{GT}Y: X correct, Y wrong (threshold 0.40); (a) 1 = same share as in the class",
                 ha="right", va="top", color=MUTED, fontsize=FONT_SIZE - 1)
    return fig


def save_disagreement_figures(results_dir: Path, out_dirs: list[Path]) -> list[Path]:
    concepts = pd.read_csv(results_dir / "concepts.csv")
    sets = pd.read_csv(results_dir / "sets.csv")
    epi = pd.read_csv(results_dir / "epithelium.csv")
    written = []
    for out_dir in out_dirs:
        for stem, pairs in FIGURES.items():
            save_figure(disagreement_figure(concepts, sets, epi, pairs), out_dir, stem)
            written += [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    print("\n".join(map(str, save_disagreement_figures(args.results_dir, [out_dir]))))


if __name__ == "__main__":
    main()
