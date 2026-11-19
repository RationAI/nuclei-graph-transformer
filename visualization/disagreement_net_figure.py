# ruff: noqa: RUF002  (mathematical symbols in text are intended)
"""Net disagreement figures (res_disagreement_context, res_disagreement_sources), from net.csv.

Saved by postprocessing.disagreement_net (run cdaf1247). One panel per pair X vs Y, side by side.
Per column (dominant concept A-H, other; benign epithelial / non-epithelial; all):
Δ = (n(X>Y) − n(Y>X)) / n in percentage points (carcinoma: recall difference, filled circle;
benign: specificity difference, open square), 95 % stratified slide-bootstrap CI; zero line;
positive = X better. Under the x labels: nuclei per column and class (carcinoma, benign).

Usage:
    uv run -m visualization.disagreement_net_figure [--results-dir DIR] [--out-dir DIR] [--suffix S]
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
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/disagreement"
)
FIGURES = {  # stem -> pairs (one panel each, side by side)
    "res_disagreement_context": ("knn_vs_self", "dense_vs_knn"),
    "res_disagreement_sources": ("texture_vs_shape", "shape_vs_position"),
}
TITLES = {
    "texture_vs_shape": "Q4: texture vs shape",
    "shape_vs_position": "Q4: shape vs position only",
    "knn_vs_self": "Q3: k-NN vs self-only",
    "dense_vs_knn": "Q3: dense vs k-NN",
}
FIG_H_IN = 2.6
MUTED = "#6b6b6b"
# carcinoma: the thesis's positive-class red (intro figures); benign: the figures' blue
CLASS_STYLE = {"carcinoma": ("#d62728", "o", True), "benign": ("#2C6E9B", "s", False)}
DX = {"carcinoma": -0.17, "benign": 0.17}
XLABELS = {"benign epithelial": "benign\nepith.", "benign non-epithelial": "benign\nnon-ep.",
           "all": "all"}
GAP, SUMMARY_STEP = 0.5, 1.45  # space before the summary columns; their spacing
AXES = ((0.1, 0.29, 0.385, 0.57), (0.6, 0.29, 0.385, 0.57))
ROW_Y = (-0.215, -0.29)  # rows under the axis (axes fraction): carcinoma n, benign n


def count_label(n: float) -> str:
    if not np.isfinite(n):
        return ""
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"


def panel(ax: plt.Axes, table: pd.DataFrame, pair: str, first: bool) -> None:
    part = table[table["pair"] == pair]
    columns = list(dict.fromkeys(part["column"]))
    n_groups = sum(c not in XLABELS for c in columns)
    x = np.array([i if i < n_groups else n_groups - 1 + GAP + SUMMARY_STEP * (i - n_groups + 1)
                  for i in range(len(columns))], float)
    ax.axhline(0, color=INK, lw=0.7, zorder=1)
    ax.axvline((x[n_groups - 1] + x[n_groups]) / 2, color=GRID, lw=0.8, zorder=0)
    for cls, (color, marker, filled) in CLASS_STYLE.items():
        rows = part[part["true_class"] == cls].set_index("column")
        xs, v, lo, hi = [], [], [], []
        for xi, col in zip(x, columns, strict=True):
            if col in rows.index:
                r = rows.loc[col]
                xs.append(xi + DX[cls])
                v.append(r["delta_pp"])
                lo.append(r["delta_pp"] - r["ci_lo"])
                hi.append(r["ci_hi"] - r["delta_pp"])
        ax.errorbar(xs, v, yerr=[lo, hi], fmt=marker, ms=3.0, color=color,
                    mfc=color if filled else "white", mew=0.8, elinewidth=0.9,
                    capsize=1.6, capthick=0.9, zorder=4)
    ax.set_xticks(x, [XLABELS.get(c, c) for c in columns])
    ax.tick_params(axis="x", length=0, pad=3, labelsize=FONT_SIZE - 1)
    for label, col in zip(ax.get_xticklabels(), columns, strict=True):
        if col in XLABELS and col != "all":
            label.set_fontsize(FONT_SIZE - 2.5)
            label.set_linespacing(0.9)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    lo_all = (part["ci_lo"]).min()
    hi_all = (part["ci_hi"]).max()
    pad = 0.08 * (hi_all - lo_all)
    ax.set_ylim(min(lo_all - pad, -1), max(hi_all + pad, 1))
    x_name = part["x"].iloc[0]
    ax.set_ylabel(f"$\\Delta$ (percentage points)\npositive: {x_name} better",
                  fontsize=FONT_SIZE - 0.5)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    clean_spines(ax)
    ax.set_title(TITLES[pair], fontsize=FONT_SIZE, loc="left", pad=4)
    under = blended_transform_factory(ax.transData, ax.transAxes)
    for y, cls in zip(ROW_Y, ("carcinoma", "benign"), strict=True):
        rows = part[part["true_class"] == cls].set_index("column")["n"]
        if first:
            ax.text(x[0] - 0.75, y, "carc." if cls == "carcinoma" else "benign",
                    transform=under, ha="right", va="center", color=CLASS_STYLE[cls][0],
                    fontsize=FONT_SIZE - 2.5)
        for xi, col in zip(x, columns, strict=True):
            ax.text(xi, y, count_label(rows.get(col, np.nan)), transform=under, ha="center",
                    va="center", color=MUTED, fontsize=FONT_SIZE - 2.5)


def net_figure(table: pd.DataFrame, pairs: tuple[str, ...]) -> plt.Figure:
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(TEXT_W_IN, FIG_H_IN))
        for i, (pair, rect) in enumerate(zip(pairs, AXES, strict=True)):
            panel(fig.add_axes(rect), table, pair, first=i == 0)
        handles = [
            Line2D([], [], color=CLASS_STYLE["carcinoma"][0], marker="o", lw=0, ms=3.0),
            Line2D([], [], color=CLASS_STYLE["benign"][0], marker="s", lw=0, ms=3.0,
                   mfc="white", mew=0.8),
        ]
        fig.legend(handles, ["carcinoma: recall difference", "benign: specificity difference"],
                   loc="upper left", bbox_to_anchor=(0.09, 1.0), frameon=False, ncols=2,
                   fontsize=FONT_SIZE - 1, handletextpad=0.2)
        fig.text(0.99, 0.985, "threshold 0.40; nuclei per column under the labels",
                 ha="right", va="top", color=MUTED, fontsize=FONT_SIZE - 1)
    return fig


def save_net_figures(results_dir: Path, out_dirs: list[Path], suffix: str = "_net") -> list[Path]:
    table = pd.read_csv(results_dir / "net.csv")
    written = []
    for out_dir in out_dirs:
        for stem, pairs in FIGURES.items():
            save_figure(net_figure(table, pairs), out_dir, f"{stem}{suffix}")
            written += [out_dir / f"{stem}{suffix}.pdf", out_dir / f"{stem}{suffix}.png"]
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--suffix", default="_net")
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    print("\n".join(map(str, save_net_figures(args.results_dir, [out_dir], args.suffix))))


if __name__ == "__main__":
    main()
