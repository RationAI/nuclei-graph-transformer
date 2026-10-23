"""Figures from the existing concept morphometry outputs (no MLflow run).

Nothing is recomputed but the statistics drawn.

1. The activation curves of `xai.concept_morphometry` with an "inactive" point (kappa_j = 0,
   median and quartiles) at the left: the inactive rows are added to
   `activation_curves.csv` if it lacks them, and `figures/curves_*` are redrawn.
2. Per concept in `violin_concepts`, `figures/violins_<letter>.pdf/png`: one panel per
   measurement in VIOLIN_MEASUREMENTS, violins for the groups inactive (kappa_j = 0), the
   active tertiles weak / medium / strong (cut-offs from all of the concept's active nuclei)
   and the concept's `top_n` most active nuclei. Polygon measures use all nuclei, OD the
   random sample (the top nuclei: all OD-measured). Shapes and statistics use every nucleus
   of a group: the density is a Gaussian-smoothed fine histogram (Scott's bandwidth), each
   violin scaled to the same width; the inner box spans the quartiles, the white line is the
   median; n under each group. The y range of a measure is shared by all concepts: from the
   lowest 1st to the highest 99th percentile of the groups (the tails beyond are not drawn).
   Statistics: `violin_stats.csv`.

Concepts are named by their review letter only.
"""

from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from scipy.ndimage import gaussian_filter1d

from xai.concept_morphometry import (
    FIG_W_CM,
    MEASUREMENTS,
    THESIS_RC,
    concept_letters,
    grid_figure,
    load_split,
    named_figure,
    summary_row,
)


VIOLIN_MEASUREMENTS = ["area_um2", "long_axis_um", "short_axis_um", "circularity", "h_od_mean_norm"]
GROUPS = {  # name -> (legend label, colour)
    "inactive": (r"inactive ($\kappa_j = 0$)", "#b4b4b4"),
    "weak": ("weak", "#a9c6ec"),
    "medium": ("medium", "#5b8fd3"),
    "strong": ("strong", "#1F5FA8"),
    "top": ("top {n:,}", "#0b2f5e"),
}
TITLES = {"area_um2": r"area ($\mu\mathrm{m}^2$)", "long_axis_um": r"long axis ($\mu\mathrm{m}$)",
          "short_axis_um": r"short axis ($\mu\mathrm{m}$)", "circularity": "circularity",
          "h_od_mean_norm": "H-OD (norm.)"}
DENSITY_BINS = 512


def groups_of(kappa: NDArray, population: NDArray, top_n: int) -> dict[str, NDArray[np.bool_]]:
    active = kappa > 0
    low, high = np.quantile(kappa[active], [1 / 3, 2 / 3])
    top = np.zeros(len(kappa), dtype=bool)
    order = np.argpartition(-kappa, top_n)[:top_n]
    top[order[kappa[order] > 0]] = True
    return {"inactive": ~active & population,
            "weak": active & (kappa <= low) & population,
            "medium": active & (kappa > low) & (kappa <= high) & population,
            "strong": active & (kappa > high) & population,
            "top": top}


def group_stats(values: NDArray) -> dict:
    q01, q25, q50, q75, q99 = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {"n": len(values), "median": q50, "q25": q25, "q75": q75, "q01": q01, "q99": q99}


def density(values: NDArray, low: float, high: float) -> tuple[NDArray, NDArray]:
    """Gaussian KDE (Scott's bandwidth) of all values, on a fine grid over [low, high]."""
    counts, edges = np.histogram(values, bins=DENSITY_BINS, range=(low, high))
    width = edges[1] - edges[0]
    bandwidth = 1.06 * np.std(values) * len(values) ** (-1 / 5)
    smooth = gaussian_filter1d(counts.astype(np.float64), max(bandwidth / width, 0.5), mode="constant")
    return (edges[:-1] + edges[1:]) / 2, smooth


def count_label(n: int) -> str:
    if n >= 1e7:
        return f"{n / 1e6:.0f}M"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    return f"{n:,}"


def violin_figure(letter: str, data: dict, stats: pd.DataFrame, ranges: dict, top_n: int) -> plt.Figure:
    side = FIG_W_CM / 2.54
    fig, axes = plt.subplots(1, len(VIOLIN_MEASUREMENTS), figsize=(side, side * 0.3))
    for ax, column in zip(axes, VIOLIN_MEASUREMENTS, strict=True):
        low, high = ranges[column]
        for x, (group, (_, colour)) in enumerate(GROUPS.items()):
            values = data[column][group]
            s = stats[(stats["concept"] == letter) & (stats["measurement"] == column)
                      & (stats["group"] == group)].iloc[0]
            grid, dens = density(values, low, high)
            half = 0.4 * dens / dens.max()
            ax.fill_betweenx(grid, x - half, x + half, color=colour, lw=0, alpha=0.85)
            ax.add_patch(plt.Rectangle((x - 0.06, s["q25"]), 0.12, s["q75"] - s["q25"],
                                       facecolor="black", edgecolor="none", zorder=3))
            ax.plot([x - 0.06, x + 0.06], [s["median"]] * 2, color="white", lw=0.9, zorder=4)
            ax.text(x, -0.03, count_label(int(s["n"])), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5)
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(GROUPS) - 0.4)
        ax.set_ylim(low, high)
        ax.set_title(TITLES[column], fontsize=7, pad=3)
        ax.tick_params(axis="y", labelsize=6, length=2, width=0.4, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _, colour in GROUPS.values()]
    labels = [label.format(n=top_n) for label, _ in GROUPS.values()]
    fig.legend(handles, labels, loc="lower center", ncols=len(GROUPS), fontsize=6.5, frameon=False,
               handlelength=1.0, handleheight=0.8, columnspacing=1.2, borderaxespad=0.1)
    fig.suptitle(f"Concept {letter}", fontsize=8, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.83, wspace=0.42)
    return fig


def population_stats(morphometry: pd.DataFrame) -> pd.DataFrame:
    """Median and 5th / 95th percentile of each measure over all nuclei (OD: the random sample)."""
    random = morphometry["od_random_sample"].to_numpy()
    rows = []
    for column in VIOLIN_MEASUREMENTS:
        values = morphometry[column].to_numpy(dtype=np.float64)
        values = values[random] if MEASUREMENTS[column][1] else values
        q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
        rows.append({"measurement": column, "n": len(values), "median": q50, "q05": q05, "q95": q95})
    return pd.DataFrame(rows).set_index("measurement")


def group_handles(top_n: int, extra: list | None = None) -> tuple[list, list]:
    handles = [mpl.lines.Line2D([], [], color=colour, marker="o", ms=3, lw=0.8) for _, colour in GROUPS.values()]
    labels = [label.format(n=top_n) for label, _ in GROUPS.values()]
    for handle, label in extra or []:
        handles.append(handle)
        labels.append(label)
    return handles, labels


def points_figure(stats: pd.DataFrame, population: pd.DataFrame, letters: list[str], top_n: int,
                  relative: bool) -> plt.Figure:
    """Measures stacked vertically; per concept the groups' medians with IQR bars.

    `relative`: as % difference from the concept's inactive median (0 % line, +-5 % band).
    Otherwise the y range is the 5th-95th percentile of all nuclei; a median beyond it is
    marked by an arrow at the edge with its value.
    """
    side = FIG_W_CM / 2.54
    fig, axes = plt.subplots(len(VIOLIN_MEASUREMENTS), 1, figsize=(side, side * 1.12), sharex=True)
    offsets = np.linspace(-0.3, 0.3, len(GROUPS))
    for ax, column in zip(axes, VIOLIN_MEASUREMENTS, strict=True):
        s = stats[stats["measurement"] == column].set_index(["concept", "group"])
        spans, off_scale = [], []
        for i, letter in enumerate(letters):
            base = s.loc[(letter, "inactive"), "median"]
            for offset, (group, (_, colour)) in zip(offsets, GROUPS.items(), strict=True):
                q25, q50, q75 = s.loc[(letter, group), ["q25", "median", "q75"]]
                if relative:
                    q25, q50, q75 = (100 * (q / base - 1) for q in (q25, q50, q75))
                spans += [q25, q75]
                ax.plot([i + offset] * 2, [q25, q75], color=colour, lw=0.8, solid_capstyle="butt")
                ax.plot(i + offset, q50, "o", color=colour, ms=3, mec="none")
                if not relative:
                    off_scale.append((i + offset, q50, colour))
        if relative:
            ax.axhspan(-5, 5, color="#e6e6e6", lw=0, zorder=0)
            ax.axhline(0, color="black", lw=0.6)
            pad = 0.05 * (max(spans) - min(spans))
            ax.set_ylim(min(min(spans), -5) - pad, max(max(spans), 5) + pad)
            ax.set_ylabel(TITLES[column] + "\n" + r"$\Delta$ vs. inactive (%)", fontsize=7)
        else:
            ax.axhline(population.loc[column, "median"], color="black", lw=0.6, ls=(0, (3, 2)))
            low, high = population.loc[column, "q05"], population.loc[column, "q95"]
            ax.set_ylim(low, high)
            ax.set_ylabel(TITLES[column], fontsize=7)
            # medians beyond the 5th-95th percentile range: an arrow at the edge with the value
            for x, q50, colour in off_scale:
                if low <= q50 <= high:
                    continue
                below = q50 < low
                edge = low if below else high
                ax.plot(x, edge, "v" if below else "^", color=colour, ms=4, mec="none", clip_on=False, zorder=5)
                ax.annotate(f"{q50:.2f}" if abs(q50) < 10 else f"{q50:.0f}", (x, edge),
                            xytext=(3, 4 if below else -4), textcoords="offset points", ha="left",
                            va="bottom" if below else "top", fontsize=5.5, color=colour)
        for i in range(1, len(letters)):
            ax.axvline(i - 0.5, color="#e6e6e6", lw=0.4, zorder=0)
        ax.set_xlim(-0.5, len(letters) - 0.5)
        ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    axes[-1].set_xticks(range(len(letters)), letters, fontsize=8)
    axes[-1].tick_params(axis="x", length=0, pad=3)
    axes[-1].set_xlabel("concept", fontsize=8)
    if relative:
        extra = [(mpl.lines.Line2D([], [], color="black", lw=0.6), r"$0\,\%$"),
                 (plt.Rectangle((0, 0), 1, 1, color="#e6e6e6"), r"$\pm 5\,\%$")]
    else:
        extra = [(mpl.lines.Line2D([], [], color="black", lw=0.6, ls=(0, (3, 2))), "median of all nuclei")]
    handles, labels = group_handles(top_n, extra)
    fig.legend(handles, labels, loc="upper center", ncols=len(labels), fontsize=6.5, frameon=False,
               handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.05, top=0.955, hspace=0.12)
    return fig


# heatmap classes: |delta| below the first threshold = no difference; then 3 classes, lower
# (blue) or higher (orange); thresholds per unit (%, percentage points, um: 5 / 15 / 30;
# Delta of an index in [-1, 1]: 0.05 / 0.15 / 0.30)
CLASS_EDGES = {"%": [5.0, 15.0, 30.0], "pp": [5.0, 15.0, 30.0], "um": [5.0, 15.0, 30.0], "Δ": [0.05, 0.15, 0.30]}
UNIT_HEADER = {"%": "(%)", "pp": "(pp)", "um": r"($\mu\mathrm{m}$)", "Δ": r"($\Delta$ index)"}
LOWER_COLOURS = ["#c6dbef", "#6b9bd1", "#1F5FA8"]  # small, medium, large
HIGHER_COLOURS = ["#fdd8b0", "#f3a35a", "#d46a10"]
NEUTRAL_COLOUR = "#ffffff"
GRID_COLOUR = "#d9d9d9"
# cmr10 is TeX-encoded (OT1): U+2013 is missing and "<", ">" print as inverted marks, so the en
# dash uses the slot of "{" and the comparison signs are set as math
EN_DASH = "{"


def delta_class(value: float, unit: str) -> tuple[str, bool]:
    """Fill colour of a difference and whether its text should be white."""
    level = int(np.searchsorted(CLASS_EDGES[unit], abs(value), side="right"))  # 0: no difference
    if level == 0:
        return NEUTRAL_COLOUR, False
    colours = HIGHER_COLOURS if value > 0 else LOWER_COLOURS
    return colours[level - 1], level == 3


def cell_text(value: float, unit: str) -> str:
    suffix = {"%": r"\,\%", "pp": r"\,\mathrm{pp}", "um": r"\,\mu\mathrm{m}", "Δ": ""}[unit]
    digits = 2 if unit == "Δ" else 0
    if round(value, digits) == 0:  # neither "+0" nor "-0"
        return rf"$0{suffix}$"
    return rf"${value:+.{digits}f}{suffix}$"


RELATIVE_TITLES = {
    "area_um2": r"Area ($\mu\mathrm{m}^2$)",
    "long_axis_um": r"Long axis ($\mu\mathrm{m}$)",
    "short_axis_um": r"Short axis ($\mu\mathrm{m}$)",
    "circularity": r"Circularity ($4\pi A/P^2$)",
    "h_od_mean_norm": "Hematoxylin OD (slide-normalised)",
}
POOL_BAND_SPREAD = 0.15  # pool the inactive IQR band if its edges vary < this share of its width
RELATIVE_CAPTION = (
    "Nuclear morphometry for the eight stable concepts A{d}H (test set). For each concept, the nuclei are "
    "grouped by activation: weak, medium and strong (tertiles of the active nuclei) and the 2,000 most strongly "
    "activated. Dots show the group median as a percentage difference from the median of nuclei where the concept "
    "is inactive; bars show the group{a}s interquartile range on the same scale (a spread, not a confidence "
    "interval). The grey band is the interquartile range of the inactive nuclei. +100 % means twice the inactive "
    "median."
).format(d="\u2013", a="\u2019")


def percent_ticks(value: float, _: int) -> str:
    return "0" if abs(value) < 1e-9 else rf"${value:+.0f}\,\%$"


def delta_ticks(value: float, _: int) -> str:
    return "0" if abs(value) < 1e-9 else rf"${value:+.2g}$"


def unit_ticks(suffix: str):
    def ticks(value: float, _: int) -> str:
        return "0" if abs(value) < 1e-9 else rf"${value:+.0f}\,{suffix}$"
    return ticks


TICKS = {"%": percent_ticks, "Δ": delta_ticks, "um": unit_ticks(r"\mu\mathrm{m}"), "pp": unit_ticks(r"\mathrm{pp}")}


def relative_panel(ax, rows: pd.DataFrame, letters: list[str], unit: str, band: bool = True) -> dict:
    """One panel of active-group medians and IQRs relative to the inactive median.

    As % (`unit` "%") or as a difference in index units ("Δ"), um or percentage points ("pp"),
    over the inactive nuclei's IQR band (`band`). `rows` is indexed by (concept, group) with
    median, q25, q75 (for a share: the share and its interval). The band is pooled over the concepts
    (median of their edges) if its edges vary by less than POOL_BAND_SPREAD of its width,
    otherwise drawn per concept column. The y scale is symmetric around 0 unless one side is
    under a third of the other.
    """
    def rel(q: float, base: float) -> float:
        return 100 * (q / base - 1) if unit == "%" else q - base

    groups = [g for g in GROUPS if g != "inactive"]
    offsets = np.linspace(-0.27, 0.27, len(groups))
    inactive = rows.xs("inactive", level="group").loc[letters]
    low_edge = pd.Series([rel(q, b) for q, b in zip(inactive["q25"], inactive["median"], strict=True)], letters)
    high_edge = pd.Series([rel(q, b) for q, b in zip(inactive["q75"], inactive["median"], strict=True)], letters)
    spread = max(np.ptp(low_edge), np.ptp(high_edge)) / max(float((high_edge - low_edge).mean()), 1e-12)
    pooled = spread < POOL_BAND_SPREAD
    if not band:
        pass
    elif pooled:
        ax.axhspan(low_edge.median(), high_edge.median(), color="#e3e3e3", lw=0, zorder=0)
    else:
        for i, letter in enumerate(letters):
            ax.fill_between([i - 0.5, i + 0.5], low_edge[letter], high_edge[letter], color="#e3e3e3", lw=0, zorder=0)
    ax.axhline(0, color="black", lw=0.6, zorder=1)
    spans = [*low_edge, *high_edge] if band else []
    for i, letter in enumerate(letters):
        base = rows.loc[(letter, "inactive"), "median"]
        for offset, group in zip(offsets, groups, strict=True):
            q25, q50, q75 = (rel(q, base) for q in rows.loc[(letter, group), ["q25", "median", "q75"]])
            colour = GROUPS[group][1]
            spans += [q25, q75]
            ax.plot([i + offset] * 2, [q25, q75], color=colour, lw=0.9, solid_capstyle="butt", zorder=3)
            ax.plot(i + offset, q50, "o", color=colour, ms=3.5, mec="none", zorder=4)
    low, high = min(spans), max(spans)
    limit = max(abs(low), abs(high))
    if min(abs(low), abs(high)) >= limit / 3:
        low, high = -limit, limit
    pad = 0.06 * (high - low)
    ax.set_ylim(low - pad, high + pad)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, symmetric=low == -high, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(TICKS[unit]))
    ax.grid(axis="y", color="#ececec", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for i in range(1, len(letters)):
        ax.axvline(i - 0.5, color="#d9d9d9", lw=0.4, zorder=0)
    ax.set_xlim(-0.5, len(letters) - 0.5)
    ax.tick_params(labelsize=7, length=2, width=0.4, pad=1.5)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    return {"band": band, "pooled": bool(pooled), "edge_spread_share": round(float(spread), 3),
            "low": round(float(low_edge.median()), 3), "high": round(float(high_edge.median()), 3)}


def relative_panels_figure(panels: list[dict], letters: list[str], top_n: int, ylabel: str,
                           band_label: str = "IQR of inactive nuclei") -> tuple[plt.Figure, dict]:
    """Panels stacked vertically (`panels`: title, rows, unit, optional note and band), legend above."""
    side = FIG_W_CM / 2.54
    fig, axes = plt.subplots(len(panels), 1, figsize=(side, side * 0.24 * len(panels) + 0.75), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    bands = {}
    for ax, panel in zip(axes, panels, strict=True):
        bands[panel["title"]] = relative_panel(ax, panel["rows"], letters, panel["unit"], panel.get("band", True))
        ax.set_title(panel["title"], fontsize=8, pad=3, loc="left")
        if panel.get("note"):
            ax.text(1.0, 1.02, panel["note"], transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
    axes[-1].set_xticks(range(len(letters)), letters, fontsize=8)
    fig.supylabel(ylabel, fontsize=8, x=0.01)
    groups = [g for g in GROUPS if g != "inactive"]
    handles = [mpl.lines.Line2D([], [], color=GROUPS[g][1], marker="o", ms=3.5, lw=0.9) for g in groups]
    labels = [GROUPS[g][0].format(n=top_n) for g in groups]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#e3e3e3"))
    labels.append(band_label)
    height = fig.get_size_inches()[1]
    fig.legend(handles, labels, loc="upper center", ncols=len(labels), fontsize=7, frameon=False,
               handlelength=1.4, columnspacing=1.2, bbox_to_anchor=(0.53, 1 - 0.04 / height))
    fig.text(0.53, 1 - 0.3 / height, f"weak / medium / strong = activation tertiles of the active nuclei; top "
             f"{top_n:,} = most strongly activated nuclei", ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.3 / height, top=1 - 0.68 / height, hspace=0.38)
    return fig, bands


def relative_figure(stats: pd.DataFrame, letters: list[str], top_n: int) -> tuple[plt.Figure, dict]:
    """The 5 morphometric measures as % difference from the inactive median."""
    panels = [{"title": RELATIVE_TITLES[c], "unit": "%",
               "rows": stats[stats["measurement"] == c].set_index(["concept", "group"])}
              for c in VIOLIN_MEASUREMENTS]
    return relative_panels_figure(panels, letters, top_n, "difference from inactive nuclei (median, %)")


def percent_table(stats: pd.DataFrame, letters: list[str]) -> pd.DataFrame:
    """% difference of the top nuclei's median vs. the inactive median, concepts x measures."""
    s = stats.set_index(["concept", "measurement", "group"])["median"]
    return pd.DataFrame({column: [100 * (s[(letter, column, "top")] / s[(letter, column, "inactive")] - 1)
                                  for letter in letters] for column in VIOLIN_MEASUREMENTS}, index=letters)


def heatmap_figure(table: pd.DataFrame, titles: list[str], units: list[str], top_n: int,
                   definition: str | None = None) -> plt.Figure:
    """Differences top vs. inactive (concepts x measures) in 7 classes, the value in each cell.

    `units` per column ("%", "pp", "um" or "Δ"), shown in the column header; vector cells, no
    colour bar.
    """
    side = FIG_W_CM / 2.54
    base = side * 0.66
    extra = 0.13 * (definition.count("\n") + 1) + 0.12 if definition else 0.0  # inches, for the definition
    height = base + extra
    fig = plt.figure(figsize=(side, height))
    ax = fig.add_axes([0.08, (0.22 * base + extra) / height, 0.9, 0.64 * base / height])
    n_rows, n_cols = table.shape
    for (i, k), value in np.ndenumerate(table.to_numpy()):
        colour, dark = delta_class(value, units[k])
        ax.add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1, facecolor=colour, edgecolor=GRID_COLOUR, lw=0.4))
        ax.text(k, i, cell_text(value, units[k]), ha="center", va="center", fontsize=7,
                color="white" if dark else "black")
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    headers = [f"{title}\n{UNIT_HEADER[unit]}" for title, unit in zip(titles, units, strict=True)]
    ax.set_xticks(range(n_cols), headers, fontsize=6.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_rows), table.index, fontsize=8)
    ax.tick_params(length=0)
    ax.set_ylabel("concept", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(f"Median of the {top_n:,} most strongly activated nuclei relative to nuclei where the "
                 "concept is inactive", fontsize=8, y=1 - 0.1 / height)
    if definition:
        fig.text(0.03, (extra - 0.06) / height, definition, ha="left", va="top", fontsize=6, linespacing=1.3)

    # legend: 7 labelled patches, lower (blue, large first) | no difference | higher (orange)
    legend = fig.add_axes([0.08, (0.01 * base + extra) / height, 0.9, 0.16 * base / height])
    legend.set_xlim(0, 7)
    legend.set_ylim(0, 1)
    legend.axis("off")
    used = [u for u in CLASS_EDGES if u in units]
    if used == ["Δ"] or all(u != "Δ" for u in used):
        a, b, c = CLASS_EDGES[used[0]]
        labels = [rf"$>{c:g}$", f"{b:g}{EN_DASH}{c:g}", f"{a:g}{EN_DASH}{b:g}", rf"$<{a:g}$ (no difference)",
                  f"{a:g}{EN_DASH}{b:g}", f"{b:g}{EN_DASH}{c:g}", rf"$>{c:g}$"]
    else:  # % / pp / um and Delta index together
        labels = [r"$>30$" + "\n" + r"($>0.30$)", f"15{EN_DASH}30\n(0.15{EN_DASH}0.30)",
                  f"5{EN_DASH}15\n(0.05{EN_DASH}0.15)", r"$<5$ ($<0.05$)" + "\nno difference",
                  f"5{EN_DASH}15\n(0.05{EN_DASH}0.15)", f"15{EN_DASH}30\n(0.15{EN_DASH}0.30)",
                  r"$>30$" + "\n" + r"($>0.30$)"]
    colours = [*LOWER_COLOURS[::-1], NEUTRAL_COLOUR, *HIGHER_COLOURS]
    for x, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        legend.add_patch(plt.Rectangle((x + 0.3, 0.6), 0.4, 0.18, facecolor=colour, edgecolor=GRID_COLOUR, lw=0.4))
        legend.text(x + 0.5, 0.53, label, ha="center", va="top", fontsize=6, linespacing=1.15)
    legend.text(1.5, 0.86, "lower than in nuclei where the concept is inactive", ha="center", va="bottom",
                fontsize=6.5)
    legend.text(5.5, 0.86, "higher than in nuclei where the concept is inactive", ha="center", va="bottom",
                fontsize=6.5)
    units_note = ", ".join({"%": "%", "pp": "percentage points", "um": r"$\mu$m"}[u] for u in used if u != "Δ")
    note = f"class limits in {units_note}" if units_note else ""
    if "Δ" in used:
        note += ("; " if note else "") + r"in brackets: $\Delta$ index" if units_note else r"$\Delta$ index"
    legend.text(3.5, -0.08, note, ha="center", va="bottom", fontsize=6)
    return fig


def counts_table(stats: pd.DataFrame, letters: list[str], top_n: int) -> tuple[pd.DataFrame, str]:
    """N per group: polygon measures (all nuclei) and OD (random sample)."""
    rows = []
    for letter in letters:
        row = {"concept": letter}
        for kind, column in (("polygon", "area_um2"), ("od", "h_od_mean_norm")):
            s = stats[(stats["concept"] == letter) & (stats["measurement"] == column)].set_index("group")["n"]
            row |= {f"{kind}_{group}": int(s[group]) for group in GROUPS}
        rows.append(row)
    table = pd.DataFrame(rows)
    names = [label.format(n=top_n).replace(r"($\kappa_j = 0$)", "").strip() for label, _ in GROUPS.values()]
    lines = ["\\begin{tabular}{l " + "r " * len(GROUPS) + "r " * len(GROUPS) + "}", "\\toprule",
             (f" & \\multicolumn{{{len(GROUPS)}}}{{c}}{{polygon measures (all nuclei)}} & "
             f"\\multicolumn{{{len(GROUPS)}}}{{c}}{{H-OD (random sample)}} \\\\"),
             f"\\cmidrule(lr){{2-{1 + len(GROUPS)}}} \\cmidrule(lr){{{2 + len(GROUPS)}-{1 + 2 * len(GROUPS)}}}",
             "Concept & " + " & ".join(names * 2) + " \\\\", "\\midrule"]
    for _, r in table.iterrows():
        cells = [f"{r[f'{kind}_{g}']:,}".replace(",", "\\,") for kind in ("polygon", "od") for g in GROUPS]
        lines.append(r["concept"] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return table, "\n".join(lines)


def inactive_rows(codes: NDArray, morphometry: pd.DataFrame, concepts: list[tuple[int, str]]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The curves' inactive rows and the reference medians (all / inactive)."""
    random = morphometry["od_random_sample"].to_numpy()
    rows, medians = [], []
    for j, letter in concepts:
        kappa = np.asarray(codes[:, j], dtype=np.float64)
        for column, (_, od, _) in MEASUREMENTS.items():
            values = morphometry[column].to_numpy(dtype=np.float64)
            population = random if od else np.ones(len(kappa), dtype=bool)
            off = (kappa == 0) & population
            rows.append({"concept": letter, "measurement": column, **summary_row("inactive", kappa[off], values[off])})
            medians.append({"concept": letter, "measurement": column, "median_all": float(np.median(values[population])),
                            "median_inactive": float(np.median(values[off]))})
    return pd.DataFrame(rows), pd.DataFrame(medians)


@with_cli_args(["+xai=concept_morphometry"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
def main(config: DictConfig) -> None:
    out = Path(config.morphometry_dir)
    concepts = concept_letters(config.deck_key)
    letters = [letter for _, letter in concepts]
    codes, morphometry, _ = load_split(config, "test", [*MEASUREMENTS, "od_random_sample"])

    # 1. curves with the inactive point
    curves = pd.read_csv(out / "activation_curves.csv", dtype={"bin": str})
    spearman = pd.read_csv(out / "spearman.csv")
    inactive, medians = inactive_rows(codes, morphometry, concepts)
    if not (curves["bin"] == "inactive").any():
        curves = pd.concat([inactive, curves], ignore_index=True)
        order = {letter: i for i, letter in enumerate(letters)}
        curves = curves.sort_values(["concept", "measurement"], key=lambda s: s.map(order) if s.name == "concept"
                                    else s.map({c: i for i, c in enumerate(MEASUREMENTS)}), kind="stable")
        curves.to_csv(out / "activation_curves.csv", index=False)
    with mpl.rc_context(THESIS_RC):
        for column in MEASUREMENTS:
            fig = grid_figure(column, curves, spearman, medians, letters, config.n_bins)
            fig.savefig(out / "figures" / f"curves_{column}.pdf")
            fig.savefig(out / "figures" / f"curves_{column}.png", dpi=200)
            plt.close(fig)
        fig = named_figure(list(config.named_concepts), curves, spearman, medians, config.n_bins)
        fig.savefig(out / "figures" / "curves_named_concepts.pdf")
        fig.savefig(out / "figures" / "curves_named_concepts.png", dpi=200)
        plt.close(fig)

    # 2. violins
    random = morphometry["od_random_sample"].to_numpy()
    index = {letter: j for j, letter in concepts}
    data, stats = {}, []
    for letter in config.violin_concepts:
        kappa = np.asarray(codes[:, index[letter]], dtype=np.float64)
        data[letter] = {}
        for column in VIOLIN_MEASUREMENTS:
            values = morphometry[column].to_numpy(dtype=np.float64)
            population = random if MEASUREMENTS[column][1] else np.ones(len(kappa), dtype=bool)
            groups = groups_of(kappa, population, config.top_n)
            data[letter][column] = {g: values[mask] for g, mask in groups.items()}
            assert np.isfinite(data[letter][column]["top"]).all(), (letter, column)
            stats += [{"concept": letter, "measurement": column, "group": g, **group_stats(v)}
                      for g, v in data[letter][column].items()]
    stats = pd.DataFrame(stats)
    stats.to_csv(out / "violin_stats.csv", index=False)
    ranges = {column: (stats.loc[stats["measurement"] == column, "q01"].min(),
                       stats.loc[stats["measurement"] == column, "q99"].max()) for column in VIOLIN_MEASUREMENTS}
    with mpl.rc_context(THESIS_RC):
        for letter in config.violin_concepts:
            fig = violin_figure(letter, data[letter], stats, ranges, config.top_n)
            fig.savefig(out / "figures" / f"violins_{letter}.pdf")
            fig.savefig(out / "figures" / f"violins_{letter}.png", dpi=300)
            plt.close(fig)

    # 3. all concepts: points (absolute and relative to inactive), heatmap, n per group
    population = population_stats(morphometry)
    population.to_csv(out / "population_stats.csv")
    letters_v = list(config.violin_concepts)
    with mpl.rc_context(THESIS_RC):
        fig = points_figure(stats, population, letters_v, config.top_n, False)
        fig.savefig(out / "figures" / "morphometry_groups.pdf")
        fig.savefig(out / "figures" / "morphometry_groups.png", dpi=300)
        plt.close(fig)
        fig, _ = relative_figure(stats, letters_v, config.top_n)
        fig.savefig(out / "figures" / "morphometry_groups_relative.pdf")
        fig.savefig(out / "figures" / "morphometry_groups_relative.png", dpi=300)
        plt.close(fig)
    (out / "figures" / "morphometry_relative_caption.txt").write_text(RELATIVE_CAPTION + "\n")
    # the heatmap (with the neighbourhood shape measures) is drawn by xai.concept_context_figures
    percent_table(stats, letters_v).round(2).to_csv(out / "top_vs_inactive_percent.csv", index_label="concept")
    counts, tex = counts_table(stats, letters_v, config.top_n)
    counts.to_csv(out / "group_counts.csv", index=False)
    (out / "group_counts.tex").write_text(tex)


if __name__ == "__main__":
    main()
