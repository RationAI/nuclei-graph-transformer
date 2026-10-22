"""Tissue context of the concepts' nuclei: figures for `violin_concepts` (no MLflow run).

From `context_<split>.parquet` (`xai.morphometry_context`) and the index labels, per concept
the groups of `xai.concept_morphometry_figures` (inactive, active tertiles weak / medium /
strong, the top `top_n`; all nuclei) and, per attribute:

- continuous (`knn8_mean_dist_um`, `boundary_dist_um`; NaN excluded): median and quartiles;
- shares (`stroma`, `benign_epithelium`, `cancer_epithelium` from the compartment at the
  centroid; `carcinoma_label`): the share of the group's nuclei, with a 95 % Wilson interval.

Figures (absolute; relative to the inactive group; heatmap top vs. inactive):
`context_groups`, `context_groups_relative`, `context_top_vs_inactive`, and the same for the
carcinoma labels as `label_share_*` (kept apart: not for the pathologist). Relative units:
% for the neighbour distance, um for the signed boundary distance (its median can be near or
below 0), percentage points for shares; the heatmap's colours are scaled per column.

Validation check: the top-minus-inactive differences and, for the continuous attributes,
Spearman rho over the active nuclei, test vs. validation (`context_validation_check.csv`).
Test rho with a slide-bootstrap CI: `context_spearman.csv`. Concepts by letter only.
"""

from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from scipy.stats import spearmanr

from xai.concept_morphometry import FIG_W_CM, THESIS_RC, concept_letters, spearman_ci
from xai.concept_morphometry_figures import (
    GROUPS,
    TITLES,
    VIOLIN_MEASUREMENTS,
    group_handles,
    groups_of,
    percent_table,
)
from xai.concept_morphometry_figures import heatmap_figure as percent_heatmap


# name -> (label, kind, relative unit): kind "continuous" or "share"
ATTRIBUTES = {
    "knn8_mean_dist_um": (r"8-NN distance ($\mu\mathrm{m}$)", "continuous", "%"),
    "boundary_dist_um": (r"boundary distance ($\mu\mathrm{m}$)", "continuous", "um"),
    "stroma": ("stroma (%)", "share", "pp"),
    "benign_epithelium": ("benign epithelium (%)", "share", "pp"),
    "cancer_epithelium": ("cancerous epithelium (%)", "share", "pp"),
}
LABEL_ATTRIBUTES = {"carcinoma_label": ("carcinoma-labelled (%)", "share", "pp")}
# neighbourhood shape (xai.morphometry_context); also extra columns of the morphometry heatmap
SHAPE_ATTRIBUTES = {
    "irregularity": ("contour irregularity", "continuous", "%"),
    "pleomorphism_cv": ("pleomorphism (CV)", "continuous", "%"),
    "orientation_coherence": ("orientation coherence", "continuous", "%"),
}
SHAPE_HEAT_TITLES = ["contour\nirregularity", "pleomorphism\n(CV)", "orientation\ncoherence"]
RELATIVE_LABEL = {"%": r"$\Delta$ vs. inactive (%)", "um": r"$\Delta$ vs. inactive ($\mu\mathrm{m}$)",
                  "pp": r"$\Delta$ vs. inactive (pp)"}
HEAT_HEADER = {"%": "%", "um": r"$\mu\mathrm{m}$", "pp": "pp"}


def attribute_values(context: pd.DataFrame, labels: NDArray) -> dict[str, NDArray]:
    compartment = context["compartment"].to_numpy()
    return {
        "knn8_mean_dist_um": context["knn8_mean_dist_um"].to_numpy(dtype=np.float64),
        "boundary_dist_um": context["boundary_dist_um"].to_numpy(dtype=np.float64),
        **{name: context[name].to_numpy(dtype=np.float64) for name in SHAPE_ATTRIBUTES},
        "stroma": (compartment == 1).astype(np.float64),
        "benign_epithelium": (compartment == 2).astype(np.float64),
        "cancer_epithelium": (compartment >= 3).astype(np.float64),
        "carcinoma_label": labels.astype(np.float64),
    }


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def stats_row(values: NDArray, kind: str) -> dict:
    values = values[np.isfinite(values)]
    if kind == "share":
        p = float(values.mean())
        low, high = wilson(p, len(values))
        return {"n": len(values), "value": 100 * p, "low": 100 * low, "high": 100 * high}
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {"n": len(values), "value": q50, "low": q25, "high": q75}


def group_stats(codes: NDArray, values: dict, concepts: list[tuple[int, str]], letters: list[str],
                specs: dict, top_n: int) -> pd.DataFrame:
    index = {letter: j for j, letter in concepts}
    everyone = np.ones(len(codes), dtype=bool)
    rows = []
    for letter in letters:
        groups = groups_of(np.asarray(codes[:, index[letter]], dtype=np.float64), everyone, top_n)
        for name, (_, kind, _) in specs.items():
            rows += [{"concept": letter, "attribute": name, "group": g, **stats_row(values[name][mask], kind)}
                     for g, mask in groups.items()]
    for name, (_, kind, _) in specs.items():
        rows.append({"concept": "all", "attribute": name, "group": "all", **stats_row(values[name], kind)})
    return pd.DataFrame(rows)


def relative(stats: pd.DataFrame, unit: str, base: float) -> pd.DataFrame:
    out = stats.copy()
    for column in ("value", "low", "high"):
        out[column] = 100 * (stats[column] / base - 1) if unit == "%" else stats[column] - base
    return out


def points_figure(stats: pd.DataFrame, specs: dict, letters: list[str], top_n: int, as_relative: bool,
                  ranges: dict) -> plt.Figure:
    side = FIG_W_CM / 2.54
    fig, axes = plt.subplots(len(specs), 1, figsize=(side, side * 0.235 * len(specs) + 0.5), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    offsets = np.linspace(-0.3, 0.3, len(GROUPS))
    for ax, (name, (label, _, unit)) in zip(axes, specs.items(), strict=True):
        s = stats[stats["attribute"] == name].set_index(["concept", "group"])
        spans, off_scale = [], []
        for i, letter in enumerate(letters):
            rows = s.loc[letter]
            if as_relative:
                rows = relative(rows, unit, rows.loc["inactive", "value"])
            for offset, (group, (_, colour)) in zip(offsets, GROUPS.items(), strict=True):
                low, value, high = rows.loc[group, ["low", "value", "high"]]
                spans += [low, high]
                ax.plot([i + offset] * 2, [low, high], color=colour, lw=0.8, solid_capstyle="butt")
                ax.plot(i + offset, value, "o", color=colour, ms=3, mec="none")
                off_scale.append((i + offset, value, colour))
        if as_relative:
            if unit in ("%", "pp"):
                ax.axhspan(-5, 5, color="#e6e6e6", lw=0, zorder=0)
            ax.axhline(0, color="black", lw=0.6)
            pad = 0.05 * (max(spans) - min(spans))
            ax.set_ylim(min(min(spans), -5) - pad, max(max(spans), 5) + pad)
            ax.set_ylabel(label.split(" (")[0] + "\n" + RELATIVE_LABEL[unit], fontsize=7)
        else:
            ax.axhline(s.loc[("all", "all"), "value"], color="black", lw=0.6, ls=(0, (3, 2)))
            low, high = ranges[name]
            ax.set_ylim(low, high)
            ax.set_ylabel(label, fontsize=7)
            for x, value, colour in off_scale:  # medians beyond the range: an arrow at the edge
                if low <= value <= high:
                    continue
                below = value < low
                edge = low if below else high
                ax.plot(x, edge, "v" if below else "^", color=colour, ms=4, mec="none", clip_on=False, zorder=5)
                ax.annotate(f"{value:.1f}", (x, edge), xytext=(3, 4 if below else -4), textcoords="offset points",
                            ha="left", va="bottom" if below else "top", fontsize=5.5, color=colour)
        for i in range(1, len(letters)):
            ax.axvline(i - 0.5, color="#e6e6e6", lw=0.4, zorder=0)
        ax.set_xlim(-0.5, len(letters) - 0.5)
        ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    axes[-1].set_xticks(range(len(letters)), letters, fontsize=8)
    axes[-1].tick_params(axis="x", length=0, pad=3)
    axes[-1].set_xlabel("concept", fontsize=8)
    if as_relative:
        extra = [(mpl.lines.Line2D([], [], color="black", lw=0.6), "0"),
                 (plt.Rectangle((0, 0), 1, 1, color="#e6e6e6"), r"$\pm 5$ (% or pp)")]
    else:
        extra = [(mpl.lines.Line2D([], [], color="black", lw=0.6, ls=(0, (3, 2))), "all nuclei")]
    handles, labels = group_handles(top_n, extra)
    fig.legend(handles, labels, loc="upper center", ncols=len(labels), fontsize=6.5, frameon=False,
               handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)
    height = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.55 / height, top=1 - 0.3 / height, hspace=0.12)
    return fig


def top_vs_inactive(stats: pd.DataFrame, specs: dict, letters: list[str]) -> pd.DataFrame:
    s = stats.set_index(["concept", "attribute", "group"])["value"]
    table = {}
    for name, (_, _, unit) in specs.items():
        top = np.array([s[(letter, name, "top")] for letter in letters])
        base = np.array([s[(letter, name, "inactive")] for letter in letters])
        table[name] = 100 * (top / base - 1) if unit == "%" else top - base
    return pd.DataFrame(table, index=letters)


def heatmap_figure(table: pd.DataFrame, specs: dict, top_n: int) -> plt.Figure:
    """Colours scaled per column (the units differ); values with their unit in each cell."""
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blue_orange", ["#1F5FA8", "#f7f7f7", "#E08214"])
    values = table.to_numpy()
    scaled = values / np.maximum(np.abs(values).max(axis=0, keepdims=True), 1e-9)
    side = FIG_W_CM / 2.54
    fig, ax = plt.subplots(figsize=(max(side * 0.16 * len(specs) + 1.0, 2.6), side * 0.62))
    ax.imshow(scaled, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for (i, k), value in np.ndenumerate(values):
        ax.text(k, i, rf"${value:+.1f}$" if abs(value) < 10 else rf"${value:+.0f}$", ha="center", va="center",
                fontsize=7, color="white" if abs(scaled[i, k]) > 0.6 else "black")
    headers = [f"{label.split(' (')[0]}\n({HEAT_HEADER[unit]})" for label, _, unit in specs.values()]
    ax.set_xticks(range(len(specs)), headers, fontsize=6.5)
    ax.set_yticks(range(len(table)), table.index, fontsize=8)
    ax.tick_params(length=0)
    ax.set_ylabel("concept", fontsize=8)
    ax.set_title(f"top {top_n:,} vs. inactive (colour scaled per column)", fontsize=7, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def ranges_for(values: dict, specs: dict, stats: pd.DataFrame) -> dict:
    """Absolute y range: 5th-95th percentile of all nuclei (continuous), 0 to the largest CI (shares)."""
    out = {}
    for name, (_, kind, _) in specs.items():
        if kind == "continuous":
            v = values[name][np.isfinite(values[name])]
            out[name] = tuple(np.quantile(v, [0.05, 0.95]))
        else:
            out[name] = (0.0, 1.08 * stats.loc[stats["attribute"] == name, "high"].max())
    return out


def load(config: DictConfig, split: str) -> tuple[NDArray, dict]:
    out = Path(config.morphometry_dir)
    codes = np.load(Path(config.concept_dir) / "codes_test.npy" if split == "test"
                    else out / f"codes_{split}.npy", mmap_mode="r")
    context = pd.read_parquet(out / f"context_{split}.parquet")
    labels = pq.read_table(Path(config.out_dir) / "activations" / split / "index.parquet",
                           columns=["label"]).column("label").to_numpy()
    assert len(codes) == len(context) == len(labels)
    return codes, attribute_values(context, labels)


@with_cli_args(["+xai=concept_morphometry"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
def main(config: DictConfig) -> None:
    out = Path(config.morphometry_dir)
    concepts = concept_letters(config.deck_key)
    letters = list(config.violin_concepts)
    specs_all = ATTRIBUTES | SHAPE_ATTRIBUTES | LABEL_ATTRIBUTES

    codes, values = load(config, "test")
    stats = group_stats(codes, values, concepts, letters, specs_all, config.top_n)
    stats.to_csv(out / "context_stats.csv", index=False)
    ranges = ranges_for(values, specs_all, stats)
    with mpl.rc_context(THESIS_RC):
        for stem, specs in (("context", ATTRIBUTES), ("shape", SHAPE_ATTRIBUTES), ("label_share", LABEL_ATTRIBUTES)):
            for suffix, as_relative in (("groups", False), ("groups_relative", True)):
                fig = points_figure(stats, specs, letters, config.top_n, as_relative, ranges)
                fig.savefig(out / "figures" / f"{stem}_{suffix}.pdf")
                fig.savefig(out / "figures" / f"{stem}_{suffix}.png", dpi=300)
                plt.close(fig)
            if stem == "shape":  # its heatmap columns join the morphometry heatmap below
                continue
            table = top_vs_inactive(stats, specs, letters)
            table.round(2).to_csv(out / f"{stem}_top_vs_inactive.csv", index_label="concept")
            fig = heatmap_figure(table, specs, config.top_n)
            fig.savefig(out / "figures" / f"{stem}_top_vs_inactive.pdf")
            fig.savefig(out / "figures" / f"{stem}_top_vs_inactive.png", dpi=300)
            plt.close(fig)

        # the morphometry heatmap: the 5 shape measures and the neighbourhood shape measures (all %)
        heat = pd.concat([percent_table(pd.read_csv(out / "violin_stats.csv"), letters),
                          top_vs_inactive(stats, SHAPE_ATTRIBUTES, letters)], axis=1)
        heat.round(2).to_csv(out / "morphometry_top_vs_inactive.csv", index_label="concept")
        fig = percent_heatmap(heat, [TITLES[c] for c in VIOLIN_MEASUREMENTS] + SHAPE_HEAT_TITLES, config.top_n)
        fig.savefig(out / "figures" / "morphometry_top_vs_inactive.pdf")
        fig.savefig(out / "figures" / "morphometry_top_vs_inactive.png", dpi=300)
        plt.close(fig)

    # Spearman rho of kappa_j with the continuous attributes (active nuclei), test with CI
    slides = pd.read_parquet(config.slides)
    rng = np.random.default_rng(config.seed)
    index = {letter: j for j, letter in concepts}
    continuous = [n for n, (_, kind, _) in (ATTRIBUTES | SHAPE_ATTRIBUTES).items() if kind == "continuous"]
    test_slides = slides[slides["split"] == "test"].sort_values("offset")
    slide = np.repeat(np.arange(len(test_slides)), test_slides["n_nuclei"].to_numpy())
    rho_rows = []
    for letter in letters:
        kappa = np.asarray(codes[:, index[letter]], dtype=np.float64)
        for name in continuous:
            on = (kappa > 0) & np.isfinite(values[name])
            rho_rows.append({"concept": letter, "attribute": name,
                             **spearman_ci(kappa[on], values[name][on], slide[on], len(test_slides),
                                           config.n_bootstrap, rng)})
    rho = pd.DataFrame(rho_rows)
    rho.to_csv(out / "context_spearman.csv", index=False)

    # validation check
    codes_v, values_v = load(config, "validation")
    stats_v = group_stats(codes_v, values_v, concepts, letters, specs_all, config.top_n)
    stats_v.to_csv(out / "context_stats_validation.csv", index=False)
    check = (top_vs_inactive(stats, specs_all, letters).stack().rename("top_minus_inactive_test").to_frame()
             .join(top_vs_inactive(stats_v, specs_all, letters).stack().rename("top_minus_inactive_validation")))
    check.index.names = ["concept", "attribute"]
    rho_v = []
    for letter in letters:
        kappa = np.asarray(codes_v[:, index[letter]], dtype=np.float64)
        for name in continuous:
            on = (kappa > 0) & np.isfinite(values_v[name])
            rho_v.append({"concept": letter, "attribute": name,
                          "rho_validation": float(spearmanr(kappa[on], values_v[name][on])[0])})
    check = check.join(rho.set_index(["concept", "attribute"])["rho"].rename("rho_test"))
    check = check.join(pd.DataFrame(rho_v).set_index(["concept", "attribute"]))
    check.to_csv(out / "context_validation_check.csv")


if __name__ == "__main__":
    main()
