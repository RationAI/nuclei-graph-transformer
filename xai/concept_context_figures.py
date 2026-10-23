"""Tissue context and neighbourhood shape of the concepts' nuclei (no MLflow run).

From `context_<split>.parquet` (`xai.morphometry_context`) and the index labels, per concept
in `violin_concepts` the groups of `xai.concept_morphometry_figures` (inactive, active tertiles
weak / medium / strong, the top `top_n`; all nuclei) and, per attribute:

- continuous (8-NN distance, signed boundary distance, contour irregularity, pleomorphism of
  neighbours, orientation coherence of neighbours; NaN excluded): median and quartiles, with
  `defined_share` (orientation coherence: nuclei with axis ratio > 1.2 and such neighbours);
- shares (stroma, benign / cancerous / all epithelium at the centroid; carcinoma labels): the
  share of the group's nuclei, with a 95 % Wilson interval.

Differences top vs. inactive: % for the 8-NN distance, irregularity and pleomorphism, um for
the signed boundary distance (its median can be near or below 0), percentage points for
shares, index units ("Δ") for orientation coherence. Heatmaps in 7 classes with the unit in
each column header (limits 5 / 15 / 30, or 0.05 / 0.15 / 0.30 for Δ).

Figures in `figures/`: `context_*` (benign / cancerous epithelium apart), `shape_groups*`,
`label_share_*` (not for the pathologist) and the morphometry heatmap with the neighbourhood
shape columns. `figures/pathologist/`: letters only, epithelium merged, no label shares: the
context figures (merged), `shape_groups*`, the morphometry heatmap and copies of
`morphometry_groups*` and `context_shape_matched` with their captions.

Validation check: the top-minus-inactive differences and, for the continuous attributes,
Spearman rho over the active nuclei, test vs. validation (`context_validation_check.csv`).
Test rho with a slide-bootstrap CI: `context_spearman.csv`.
"""

from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import shutil

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
    EN_DASH,
    GROUPS,
    VIOLIN_MEASUREMENTS,
    group_handles,
    groups_of,
    percent_table,
    relative_panels_figure,
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
# for the pathologist: epithelium merged (benign + cancerous), no label shares
PATHOLOGIST_ATTRIBUTES = {
    "knn8_mean_dist_um": ATTRIBUTES["knn8_mean_dist_um"],
    "boundary_dist_um": ATTRIBUTES["boundary_dist_um"],
    "epithelium": ("epithelium (%)", "share", "pp"),
    "stroma": ATTRIBUTES["stroma"],
}
# neighbourhood shape (xai.morphometry_context); also extra columns of the morphometry heatmap.
# Orientation coherence is an index in [-1, 1]: differences in index units ("Δ"), not %
SHAPE_ATTRIBUTES = {
    "irregularity": ("contour irregularity", "continuous", "%"),
    "pleomorphism_cv": ("pleomorphism of neighbours (CV of area)", "continuous", "%"),
    "orientation_coherence": ("orientation coherence of neighbours", "continuous", "Δ"),
}
SHAPE_TITLES = {
    "irregularity": r"Contour irregularity ($1 - A/A_\mathrm{hull}$), % difference",
    "pleomorphism_cv": "Pleomorphism of neighbours (CV of the area of the 8 nearest neighbours), % difference",
    "orientation_coherence": r"Orientation coherence of neighbours (mean $\cos 2\Delta\psi$), $\Delta$ index",
}
# in the statistics only: the earlier pleomorphism with the nucleus itself
EXTRA_ATTRIBUTES = {"pleomorphism_incl_self": ("pleomorphism incl. the nucleus (CV)", "continuous", "%")}
HEAT_NAMES = {"area_um2": "area", "long_axis_um": "long axis", "short_axis_um": "short axis",
              "circularity": "circularity", "h_od_mean_norm": "H-OD", "irregularity": "contour\nirregularity",
              "pleomorphism_cv": "pleomorphism\nof neighbours",
              "orientation_coherence": "orientation\ncoherence of\nneighbours",
              "knn8_mean_dist_um": "8-NN distance", "boundary_dist_um": "boundary distance", "stroma": "stroma",
              "benign_epithelium": "benign\nepithelium", "cancer_epithelium": "cancerous\nepithelium",
              "epithelium": "epithelium\nshare", "carcinoma_label": "carcinoma-\nlabelled"}
RELATIVE_LABEL = {"%": r"$\Delta$ vs. inactive (%)", "um": r"$\Delta$ vs. inactive ($\mu\mathrm{m}$)",
                  "pp": r"$\Delta$ vs. inactive (pp)", "Δ": r"$\Delta$ vs. inactive (index)"}
# the pathologist set
NUCLEUS_COLUMNS = ["area_um2", "long_axis_um", "short_axis_um", "circularity", "irregularity", "h_od_mean_norm"]
NEIGHBOURHOOD_UNITS = {"knn8_mean_dist_um": "%", "pleomorphism_cv": "%", "orientation_coherence": "Δ",
                       "boundary_dist_um": "um", "epithelium": "pp"}
NUCLEUS_DEFINITION = (
    r"Long and short axis: $2\times$ the semi-axes of the moment ellipse." + "\n"
    r"Circularity: $4\pi A/P^2$, where $A$ is the nuclear area and $P$ its perimeter; 1 = perfect circle, lower = "
    "elongated or wavy outline." + "\n"
    r"Contour irregularity: $1 - A/A_\mathrm{hull}$, where $A_\mathrm{hull}$ is the area of the convex hull (the "
    "smallest convex outline enclosing the nucleus);" + "\n"
    "0 = no indentations, higher = notched outline. H-OD: mean hematoxylin optical density, normalised by the "
    "slide median."
)
NEIGHBOURHOOD_DEFINITION = (
    "8-NN distance: mean distance to the 8 nearest nuclei; pleomorphism of neighbours: CV of the area of the 8 "
    "nearest neighbours." + "\n"
    r"Orientation coherence of neighbours: mean $\cos 2\Delta\psi$ between the long axes of a nucleus and its 8 "
    r"nearest neighbours (axis ratio $>1.2$)." + "\n"
    f"Boundary distance: positive = in stroma, away from the epithelium{EN_DASH}stroma boundary; negative = "
    "inside epithelium. Epithelium share: benign + cancerous."
)
CONTEXT_RELATIVE = {"knn8_mean_dist_um": "%", "boundary_dist_um": "um", "epithelium": "pp"}
CONTEXT_TITLES = {
    "knn8_mean_dist_um": "8-NN distance (mean distance to the 8 nearest nuclei), % difference",
    "boundary_dist_um": (r"Boundary distance ($\mu$m; positive = in stroma, negative = inside epithelium), "
                         r"difference in $\mu$m"),
    "epithelium": "Epithelium share (benign + cancerous), difference in percentage points; bars: 95 % CI",
}
CONTEXT_CAPTION = (
    "Tissue context for the eight stable concepts A{d}H (test set), grouped as in the morphometry figure (weak, "
    "medium and strong activation tertiles of the active nuclei; the 2,000 most strongly activated). 8-NN distance: "
    "mean distance to the 8 nearest nuclei of the same slide. Boundary distance: signed distance to the "
    "epithelium{d}stroma boundary of the annotation, positive = in stroma, away from the epithelium{d}stroma boundary; "
    "negative = inside epithelium. Epithelium share: share of the group{a}s nuclei in benign or cancerous epithelium. "
    "Dots show the group median (8-NN and boundary distance) or share (epithelium) as a difference from nuclei where "
    "the concept is inactive: in percent, in {mu}m and in percentage points. Bars show the group{a}s interquartile "
    "range on the same scale (a spread), for the epithelium share its 95 % confidence interval. The grey band is the "
    "interquartile range of the inactive nuclei."
).format(d="\u2013", a="\u2019", mu="\u00b5")
SHAPE_CAPTION = (
    "Neighbourhood shape for the eight stable concepts A{d}H (test set), grouped as in the morphometry figure "
    "(weak, medium and strong activation tertiles of the active nuclei; the 2,000 most strongly activated). Contour "
    "irregularity: 1 {m} A/A_hull, where A_hull is the area of the convex hull (the smallest convex outline "
    "enclosing the nucleus); 0 = no indentations, higher = notched outline. Pleomorphism of neighbours is the coefficient of variation of "
    "the area of the 8 nearest neighbours; orientation coherence of neighbours is the mean cos 2{D}{p} between the "
    "long axes of a nucleus and its 8 nearest neighbours (1 = parallel, 0 = random, {m}1 = perpendicular), defined "
    "only for nuclei and neighbours with axis ratio > 1.2. Dots show the group median relative to the median of "
    "nuclei where the concept is inactive (as a percentage difference, or for orientation coherence as a difference "
    "in index units); bars show the group{a}s interquartile range on the same scale. The grey band is the "
    "interquartile range of the inactive nuclei."
).format(d="\u2013", m="\u2212", D="\u0394", p="\u03c8", a="\u2019")


def attribute_values(context: pd.DataFrame, labels: NDArray) -> dict[str, NDArray]:
    compartment = context["compartment"].to_numpy()
    return {
        "knn8_mean_dist_um": context["knn8_mean_dist_um"].to_numpy(dtype=np.float64),
        "boundary_dist_um": context["boundary_dist_um"].to_numpy(dtype=np.float64),
        **{name: context[name].to_numpy(dtype=np.float64) for name in SHAPE_ATTRIBUTES},
        "pleomorphism_incl_self": context["pleomorphism_incl_self"].to_numpy(dtype=np.float64),
        "stroma": (compartment == 1).astype(np.float64),
        "epithelium": (compartment >= 2).astype(np.float64),
        "benign_epithelium": (compartment == 2).astype(np.float64),
        "cancer_epithelium": (compartment >= 3).astype(np.float64),
        "carcinoma_label": labels.astype(np.float64),
    }


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def stats_row(values: NDArray, kind: str) -> dict:
    """`n`: nuclei with a defined value, `defined_share`: of the group's `n_total`."""
    counts = {"n_total": len(values)}
    values = values[np.isfinite(values)]
    counts |= {"n": len(values), "defined_share": len(values) / max(counts["n_total"], 1)}
    if kind == "share":
        p = float(values.mean())
        low, high = wilson(p, len(values))
        return counts | {"value": 100 * p, "low": 100 * low, "high": 100 * high}
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return counts | {"value": q50, "low": q25, "high": q75}


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


def panel_rows(stats: pd.DataFrame, name: str) -> pd.DataFrame:
    """An attribute's group statistics as `relative_panel` rows (median / share and its spread)."""
    rows = stats[stats["attribute"] == name].rename(columns={"value": "median", "low": "q25", "high": "q75"})
    return rows.set_index(["concept", "group"])


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
    specs_all = ATTRIBUTES | PATHOLOGIST_ATTRIBUTES | SHAPE_ATTRIBUTES | EXTRA_ATTRIBUTES | LABEL_ATTRIBUTES
    (out / "figures" / "pathologist").mkdir(parents=True, exist_ok=True)

    codes, values = load(config, "test")
    stats = group_stats(codes, values, concepts, letters, specs_all, config.top_n)
    stats.to_csv(out / "context_stats.csv", index=False)
    ranges = ranges_for(values, specs_all, stats)
    with mpl.rc_context(THESIS_RC):
        def save(fig: plt.Figure, name: str, folders: tuple[str, ...] = ("figures",)) -> None:
            for folder in folders:
                fig.savefig(out / folder / f"{name}.pdf")
                fig.savefig(out / folder / f"{name}.png", dpi=300)
            plt.close(fig)

        # internal figures (benign / cancerous epithelium apart, label shares)
        for stem, specs in (("context", ATTRIBUTES), ("label_share", LABEL_ATTRIBUTES)):
            for suffix, as_relative in (("groups", False), ("groups_relative", True)):
                save(points_figure(stats, specs, letters, config.top_n, as_relative, ranges), f"{stem}_{suffix}")
            table = top_vs_inactive(stats, specs, letters)
            table.round(2).to_csv(out / f"{stem}_top_vs_inactive.csv", index_label="concept")
            save(percent_heatmap(table, [HEAT_NAMES[c] for c in specs], [u for _, _, u in specs.values()],
                                 config.top_n), f"{stem}_top_vs_inactive")

        # neighbourhood shape: absolute, and relative in the style of the morphometry figure
        save(points_figure(stats, SHAPE_ATTRIBUTES, letters, config.top_n, False, ranges), "shape_groups")
        defined = stats[(stats["attribute"] == "orientation_coherence") & stats["concept"].isin(letters)]
        panels = [{"title": SHAPE_TITLES[name], "unit": unit, "rows": panel_rows(stats, name),
                   "note": (f"defined for {100 * defined['defined_share'].min():.0f}{EN_DASH}"
                            f"{100 * defined['defined_share'].max():.0f} % of a group's nuclei (axis ratio $>1.2$)"
                            if name == "orientation_coherence" else None)}
                  for name, (_, _, unit) in SHAPE_ATTRIBUTES.items()]
        fig, _ = relative_panels_figure(panels, letters, config.top_n, "difference from inactive nuclei (median)")
        save(fig, "shape_groups_relative")
        (out / "figures" / "shape_relative_caption.txt").write_text(SHAPE_CAPTION + "\n")

        # the morphometry heatmap: the 5 shape measures (%) and the neighbourhood shape measures
        violin_stats = pd.read_csv(out / "violin_stats.csv")
        heat = pd.concat([percent_table(violin_stats, letters), top_vs_inactive(stats, SHAPE_ATTRIBUTES, letters)],
                         axis=1)
        heat.round(3).to_csv(out / "morphometry_top_vs_inactive.csv", index_label="concept")
        units = ["%"] * len(VIOLIN_MEASUREMENTS) + [u for _, _, u in SHAPE_ATTRIBUTES.values()]
        save(percent_heatmap(heat, [HEAT_NAMES[c] for c in heat.columns], units, config.top_n),
             "morphometry_top_vs_inactive")

        # the pathologist set: letters only, epithelium merged, no stroma, no label shares
        folder = out / "figures" / "pathologist"
        for old in folder.glob("*"):
            old.unlink()
        pathologist = ("figures/pathologist",)
        nucleus = heat[NUCLEUS_COLUMNS]
        neighbourhood = pd.concat([top_vs_inactive(stats, ATTRIBUTES, letters), heat,
                                   top_vs_inactive(stats, PATHOLOGIST_ATTRIBUTES, letters)], axis=1)
        neighbourhood = neighbourhood.loc[:, ~neighbourhood.columns.duplicated()][list(NEIGHBOURHOOD_UNITS)]
        for name, table, units, definition in (
                ("nucleus_top_vs_inactive", nucleus, ["%"] * len(NUCLEUS_COLUMNS), NUCLEUS_DEFINITION),
                ("neighbourhood_top_vs_inactive", neighbourhood, list(NEIGHBOURHOOD_UNITS.values()),
                 NEIGHBOURHOOD_DEFINITION)):
            table.round(3).to_csv(out / f"{name}.csv", index_label="concept")
            save(percent_heatmap(table, [HEAT_NAMES[c] for c in table.columns], units, config.top_n, definition),
                 name, pathologist)
        panels = [{"title": CONTEXT_TITLES[name], "unit": unit, "rows": panel_rows(stats, name),
                   "band": name != "epithelium"} for name, unit in CONTEXT_RELATIVE.items()]
        fig, _ = relative_panels_figure(panels, letters, config.top_n, "difference from inactive nuclei")
        save(fig, "context_groups_relative", pathologist)
        (folder / "context_relative_caption.txt").write_text(CONTEXT_CAPTION + "\n")

    # the morphometry relative figure of xai.concept_morphometry_figures, with its caption
    for name in ("morphometry_groups_relative.pdf", "morphometry_groups_relative.png",
                 "morphometry_relative_caption.txt"):
        shutil.copy(out / "figures" / name, out / "figures" / "pathologist")

    # Spearman rho of kappa_j with the continuous attributes (active nuclei), test with CI
    slides = pd.read_parquet(config.slides)
    rng = np.random.default_rng(config.seed)
    index = {letter: j for j, letter in concepts}
    continuous = [n for n, (_, kind, _) in (ATTRIBUTES | SHAPE_ATTRIBUTES | EXTRA_ATTRIBUTES).items()
                  if kind == "continuous"]
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
