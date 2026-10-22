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


if __name__ == "__main__":
    main()
