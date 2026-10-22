"""Shape-matched control for the concepts' tissue context (no MLflow run).

For each concept in `violin_concepts`, each of its `top_n` most strongly activated test nuclei
is matched to the nearest inactive nucleus (kappa_j = 0) on z-standardised area, long axis,
short axis and circularity (means and SDs over all test nuclei): from the same slide if it
has inactive nuclei, otherwise from all inactive nuclei (with replacement). For the top and
the matched nuclei: the share in epithelium (mask >= 2) and in stroma (mask = 1), the median
8-NN distance and the median signed boundary distance, and the differences top - matched
and top - all inactive with 95 % percentile intervals from `n_bootstrap` resamples of the top
nuclei's slides (a pair stays together; the median of all inactive nuclei is kept fixed).

Writes `context_shape_matched.csv`, `context_shape_matched_balance.csv` (standardised mean
differences of the matching features, top vs. matched and vs. all inactive) and
`figures/context_shape_matched.pdf/png`. Concepts by letter only.
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
from scipy.spatial import cKDTree

from xai.concept_morphometry import FIG_W_CM, THESIS_RC, concept_letters


MATCH_FEATURES = ["area_um2", "long_axis_um", "short_axis_um", "circularity"]
# name -> (axis label, unit, kind): shares in %, differences in pp; distances: medians, um
OUTCOMES = {
    "epithelium": ("epithelium", "pp", "share"),
    "stroma": ("stroma", "pp", "share"),
    "knn8_mean_dist_um": ("8-NN distance", "um", "median"),
    "boundary_dist_um": ("boundary distance", "um", "median"),
}
COLOURS = {"inactive": "#9a9a9a", "matched": "#1F5FA8"}


def statistic(values: NDArray, kind: str) -> float:
    values = values[np.isfinite(values)]
    return float(100 * values.mean() if kind == "share" else np.median(values))


def match(z: NDArray, top: NDArray, inactive: NDArray, slide: NDArray) -> tuple[NDArray, NDArray]:
    """Nearest inactive row per top row (same slide if possible) and whether it is same-slide."""
    matched = np.empty(len(top), dtype=np.int64)
    same = np.zeros(len(top), dtype=bool)
    pool = np.flatnonzero(inactive)
    pool_slide = slide[pool]
    order = np.argsort(pool_slide, kind="stable")
    pool, pool_slide = pool[order], pool_slide[order]
    global_tree = None
    for s in np.unique(slide[top]):
        rows = np.flatnonzero(slide[top] == s)
        lo, hi = np.searchsorted(pool_slide, [s, s + 1])
        if hi > lo:
            _, k = cKDTree(z[pool[lo:hi]]).query(z[top[rows]])
            matched[rows] = pool[lo:hi][k]
            same[rows] = True
        else:
            if global_tree is None:
                global_tree = cKDTree(z[pool])
            _, k = global_tree.query(z[top[rows]])
            matched[rows] = pool[k]
    return matched, same


def bootstrap(top_values: NDArray, matched_values: NDArray, slide: NDArray, inactive_value: float, kind: str,
              n_bootstrap: int, rng: np.random.Generator) -> dict:
    slides, pair_slide = np.unique(slide, return_inverse=True)
    members = [np.flatnonzero(pair_slide == i) for i in range(len(slides))]
    diff_matched, diff_inactive = [], []
    for _ in range(n_bootstrap):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(slides), len(slides))])
        top_stat = statistic(top_values[rows], kind)
        diff_matched.append(top_stat - statistic(matched_values[rows], kind))
        diff_inactive.append(top_stat - inactive_value)
    return {"ci_low_matched": np.percentile(diff_matched, 2.5), "ci_high_matched": np.percentile(diff_matched, 97.5),
            "ci_low_inactive": np.percentile(diff_inactive, 2.5),
            "ci_high_inactive": np.percentile(diff_inactive, 97.5)}


def smd(a: NDArray, b: NDArray) -> float:
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))


def figure(table: pd.DataFrame, letters: list[str], top_n: int) -> plt.Figure:
    side = FIG_W_CM / 2.54
    fig, axes = plt.subplots(len(OUTCOMES), 1, figsize=(side, side * 0.85), sharex=True)
    for ax, (name, (label, unit, _)) in zip(axes, OUTCOMES.items(), strict=True):
        t = table[table["outcome"] == name].set_index("concept").loc[letters]
        for offset, ref in ((-0.12, "inactive"), (0.12, "matched")):
            x = np.arange(len(letters)) + offset
            ax.errorbar(x, t[f"diff_{ref}"], yerr=[t[f"diff_{ref}"] - t[f"ci_low_{ref}"],
                                                  t[f"ci_high_{ref}"] - t[f"diff_{ref}"]],
                        fmt="o", ms=3, lw=0.8, capsize=0, color=COLOURS[ref], mec="none")
        ax.axhline(0, color="black", lw=0.6)
        for i in range(1, len(letters)):
            ax.axvline(i - 0.5, color="#e6e6e6", lw=0.4, zorder=0)
        ax.set_ylabel(f"{label}\n" + (r"$\Delta$ (pp)" if unit == "pp" else r"$\Delta$ ($\mu\mathrm{m}$)"),
                      fontsize=7)
        ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    axes[-1].set_xticks(range(len(letters)), letters, fontsize=8)
    axes[-1].tick_params(axis="x", length=0, pad=3)
    axes[-1].set_xlim(-0.5, len(letters) - 0.5)
    axes[-1].set_xlabel("concept", fontsize=8)
    handles = [mpl.lines.Line2D([], [], color=c, marker="o", ms=3, lw=0.8) for c in COLOURS.values()]
    fig.legend(handles, [f"top {top_n:,} $-$ all inactive nuclei", f"top {top_n:,} $-$ shape-matched inactive nuclei"],
               loc="upper center", ncols=2, fontsize=6.5, frameon=False, borderaxespad=0.2)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.07, top=0.95, hspace=0.12)
    return fig


@with_cli_args(["+xai=concept_morphometry"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
def main(config: DictConfig) -> None:
    out = Path(config.morphometry_dir)
    concepts = {letter: j for j, letter in concept_letters(config.deck_key)}
    codes = np.load(Path(config.concept_dir) / "codes_test.npy", mmap_mode="r")
    morphometry = pd.read_parquet(out / "morphometry_test.parquet", columns=MATCH_FEATURES)
    context = pd.read_parquet(out / "context_test.parquet",
                              columns=["compartment", "knn8_mean_dist_um", "boundary_dist_um"])
    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == "test"].sort_values("offset")
    slide = np.repeat(np.arange(len(slides)), slides["n_nuclei"].to_numpy())
    assert len(codes) == len(morphometry) == len(context) == len(slide)

    features = morphometry.to_numpy(dtype=np.float64)
    z = (features - features.mean(axis=0)) / features.std(axis=0)
    compartment = context["compartment"].to_numpy()
    outcomes = {"epithelium": (compartment >= 2).astype(np.float64), "stroma": (compartment == 1).astype(np.float64),
                "knn8_mean_dist_um": context["knn8_mean_dist_um"].to_numpy(dtype=np.float64),
                "boundary_dist_um": context["boundary_dist_um"].to_numpy(dtype=np.float64)}

    rng = np.random.default_rng(config.seed)
    rows, balance = [], []
    letters = list(config.violin_concepts)
    for letter in letters:
        kappa = np.asarray(codes[:, concepts[letter]], dtype=np.float64)
        top = np.argpartition(-kappa, config.top_n)[: config.top_n]
        top = top[kappa[top] > 0]
        inactive = kappa == 0
        matched, same = match(z, top, inactive, slide)
        for k, name in enumerate(MATCH_FEATURES):
            balance.append({"concept": letter, "feature": name,
                            "smd_top_vs_inactive": smd(features[top, k], features[inactive, k]),
                            "smd_top_vs_matched": smd(features[top, k], features[matched, k])})
        for name, (_, unit, kind) in OUTCOMES.items():
            values = outcomes[name]
            top_stat, matched_stat = statistic(values[top], kind), statistic(values[matched], kind)
            inactive_stat = statistic(values[inactive], kind)
            rows.append({"concept": letter, "outcome": name, "unit": "%" if kind == "share" else unit,
                         "n_top": len(top), "n_slides": int(np.unique(slide[top]).size),
                         "same_slide_share": float(same.mean()), "unique_controls": int(np.unique(matched).size),
                         "top": top_stat, "matched": matched_stat, "inactive": inactive_stat,
                         "diff_matched": top_stat - matched_stat, "diff_inactive": top_stat - inactive_stat,
                         **bootstrap(values[top], values[matched], slide[top], inactive_stat, kind,
                                     config.n_bootstrap, rng)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "context_shape_matched.csv", index=False)
    pd.DataFrame(balance).to_csv(out / "context_shape_matched_balance.csv", index=False)
    with mpl.rc_context(THESIS_RC):
        fig = figure(table, letters, config.top_n)
        fig.savefig(out / "figures" / "context_shape_matched.pdf")
        fig.savefig(out / "figures" / "context_shape_matched.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
