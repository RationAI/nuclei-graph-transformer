"""Morphometry of the SAE concepts against their activation (steps 2 and 3).

For every concept j and measurement (polygon features of all nuclei, hematoxylin OD of the
random per-slide sample; see `xai.morphometry_features`), on the test nuclei with the
stored codes:

- activation curves: the active nuclei (kappa_j > 0) in `n_bins` quantile bins of kappa_j
  (bin edges from all active nuclei, so all measurements share them), per bin n, the kappa
  range and the median and quartiles of the measurement; plus a "top" point, the concept's
  `top_n` most strongly activated nuclei (all OD-measured); references: the median over all
  nuclei and over the inactive ones (kappa_j = 0). OD statistics use the random sample only.
- Spearman rho of kappa_j with the measurement over the active nuclei (OD: of the random
  sample), with a 95 % percentile CI from `n_bootstrap` resamples of slides. The point
  estimate is exact; the bootstrap keeps the ranks of the full sample (Pearson of fixed
  ranks over the resampled slides, from per-slide sums).
- summary table: frequency, median active vs. all, rho [CI] per measurement, and the
  measurement with the largest |rho|; a concept "tracks size" if |rho| with the area is at
  least `size_min_rho` (the axes alone can also mean elongation).

With `validation`, the polygon features' curves and rho are repeated on validation (codes
from the fixed SAE, forward pass only) and compared with test.

Blinding: concepts are named by their letter in the review deck (`deck_key`) only; nothing
written or logged names a concept's SAE index.
"""

import json
import platform
from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy.stats import rankdata

from nuclei_graph.mlflow_utils import tag_parent_run
from xai.concept_masks import code_commit


# column -> (axis label, from the OD subsample, size measurement)
MEASUREMENTS = {
    "area_um2": (r"area ($\mu\mathrm{m}^2$)", False, True),
    "perimeter_um": (r"perimeter ($\mu\mathrm{m}$)", False, True),
    "long_axis_um": (r"long axis ($\mu\mathrm{m}$)", False, True),
    "short_axis_um": (r"short axis ($\mu\mathrm{m}$)", False, True),
    "axis_ratio": ("axis ratio", False, False),
    "circularity": ("circularity", False, False),
    "h_od_mean_norm": ("hematoxylin OD (norm.)", True, False),
    "h_od_integrated_norm": ("integrated OD (norm.)", True, True),  # norm. OD x um^2
}
SHORT = {"area_um2": "Area", "perimeter_um": "Perim.", "long_axis_um": "Long ax.",
         "short_axis_um": "Short ax.", "axis_ratio": "Ratio", "circularity": "Circ.",
         "h_od_mean_norm": "OD", "h_od_integrated_norm": "Int. OD"}
LINE_COLOR = "#1F5FA8"
# the thesis font, as the other thesis figures: Computer Modern (cmr10 + mathtext cm)
THESIS_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "pdf.fonttype": 42,
}
FIG_W_CM = 16.0


def concept_letters(deck_key: str) -> list[tuple[int, str]]:
    """(SAE index, letter) in deck order; the index is used internally only."""
    key = pd.read_csv(deck_key)
    assert key["concept"].is_unique and len(key) == 16
    return list(zip(key["concept"].astype(int), key["letter"].astype(str), strict=True))


def bin_curve(kappa: NDArray, values: NDArray, bins: NDArray, n_bins: int) -> list[dict]:
    order = np.argsort(bins, kind="stable")
    bounds = np.searchsorted(bins[order], np.arange(n_bins + 1))
    rows = []
    for b in range(n_bins):
        part = order[bounds[b]: bounds[b + 1]]
        rows.append(summary_row(str(b + 1), kappa[part], values[part]))
    return rows


def summary_row(name: str, kappa: NDArray, values: NDArray) -> dict:
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75]) if len(values) else (np.nan,) * 3
    return {"bin": name, "n": len(values), "kappa_min": float(kappa.min()), "kappa_max": float(kappa.max()),
            "kappa_median": float(np.median(kappa)), "median": q50, "q25": q25, "q75": q75}


def spearman_ci(kappa: NDArray, values: NDArray, slide: NDArray, n_slides: int,
                n_bootstrap: int, rng: np.random.Generator) -> dict:
    """Exact Spearman rho and a slide-bootstrap 95 % CI (fixed ranks, per-slide sums)."""
    rx, ry = rankdata(kappa), rankdata(values)
    rx -= rx.mean()
    ry -= ry.mean()
    rho = float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))
    stats = np.stack([np.bincount(slide, weights=w, minlength=n_slides)
                      for w in (np.ones_like(rx), rx, ry, rx**2, ry**2, rx * ry)], axis=1)
    draws = rng.integers(0, n_slides, size=(n_bootstrap, n_slides))
    counts = np.stack([np.bincount(d, minlength=n_slides) for d in draws])
    n, sx, sy, sxx, syy, sxy = (counts @ stats).T
    boot = (sxy - sx * sy / n) / np.sqrt((sxx - sx**2 / n) * (syy - sy**2 / n))
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"rho": rho, "ci_low": float(low), "ci_high": float(high),
            "n_active": len(kappa), "n_slides": int(np.unique(slide).size)}


def analyse(codes: NDArray, morphometry: pd.DataFrame, slide: NDArray, concepts: list[tuple[int, str]],
            measurements: list[str], config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Curves, Spearman rho and medians (active / inactive / all) of every concept x measurement."""
    rng = np.random.default_rng(config.seed)
    n_slides = int(slide.max()) + 1
    random = morphometry["od_random_sample"].to_numpy() if "od_random_sample" in morphometry else None
    curves, spearman, medians = [], [], []
    for j, letter in concepts:
        kappa = np.asarray(codes[:, j], dtype=np.float64)
        active = kappa > 0
        edges = np.quantile(kappa[active], np.linspace(0, 1, config.n_bins + 1)[1:-1])
        bins = np.searchsorted(edges, kappa, side="right")
        top = np.argpartition(-kappa, config.top_n)[: config.top_n]
        top = top[kappa[top] > 0]
        for column in measurements:
            values = morphometry[column].to_numpy(dtype=np.float64)
            od = MEASUREMENTS[column][1]
            population = random if od else np.ones(len(kappa), dtype=bool)
            on = active & population
            rows = bin_curve(kappa[on], values[on], bins[on], config.n_bins)
            assert np.isfinite(values[top]).all(), (letter, column)
            rows.append(summary_row("top", kappa[top], values[top]))
            curves += [{"concept": letter, "measurement": column, **r} for r in rows]
            spearman.append({"concept": letter, "measurement": column, "population": "random sample" if od else "all",
                             **spearman_ci(kappa[on], values[on], slide[on], n_slides, config.n_bootstrap, rng)})
            medians.append({"concept": letter, "measurement": column, "frequency": float(active.mean()),
                            "median_active": float(np.median(values[on])),
                            "median_inactive": float(np.median(values[~active & population])),
                            "median_all": float(np.median(values[population]))})
    return pd.DataFrame(curves), pd.DataFrame(spearman), pd.DataFrame(medians)


def draw_panel(ax, curve: pd.DataFrame, med: pd.Series, n_bins: int) -> None:
    deciles = curve[curve["bin"] != "top"]
    top = curve[curve["bin"] == "top"].iloc[0]
    x = np.arange(1, n_bins + 1)
    ax.fill_between(x, deciles["q25"], deciles["q75"], color=LINE_COLOR, alpha=0.2, lw=0)
    ax.plot(x, deciles["median"], color=LINE_COLOR, lw=0.9, marker="o", ms=1.8)
    x_top = n_bins + 1.5
    ax.errorbar([x_top], [top["median"]], yerr=[[top["median"] - top["q25"]], [top["q75"] - top["median"]]],
                color=LINE_COLOR, marker="D", ms=2.2, lw=0.8, capsize=0)
    ax.axhline(med["median_all"], color="black", lw=0.6, ls=(0, (3, 2)))
    ax.axhline(med["median_inactive"], color="#7a7a7a", lw=0.6, ls=(0, (1, 1.5)))
    ax.set_xlim(0.4, x_top + 0.6)
    ax.set_xticks([1, 5, n_bins, x_top], ["1", "5", str(n_bins), "top"])
    ax.tick_params(labelsize=6, length=2, width=0.4, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def y_range(curves: pd.DataFrame, medians: pd.DataFrame) -> tuple[float, float]:
    low = min(curves["q25"].min(), medians[["median_all", "median_inactive"]].min().min())
    high = max(curves["q75"].max(), medians[["median_all", "median_inactive"]].max().max())
    pad = 0.05 * (high - low)
    return low - pad, high + pad


def grid_figure(column: str, curves: pd.DataFrame, spearman: pd.DataFrame, medians: pd.DataFrame,
                letters: list[str], n_bins: int) -> plt.Figure:
    """16 panels (deck order), shared y range, rho in each title."""
    c, s, m = (d[d["measurement"] == column] for d in (curves, spearman, medians))
    side = FIG_W_CM / 2.54
    fig, axes = plt.subplots(4, 4, figsize=(side, side * 0.92), sharex=True, sharey=True)
    for ax, letter in zip(axes.flat, letters, strict=True):
        draw_panel(ax, c[c["concept"] == letter], m[m["concept"] == letter].iloc[0], n_bins)
        rho = s[s["concept"] == letter].iloc[0]["rho"]
        ax.set_title(rf"{letter}: $\rho = {rho:.2f}$", fontsize=7, pad=2)
    axes[0, 0].set_ylim(*y_range(c, m))
    fig.supxlabel(r"activation decile of $\kappa_j$ (top: " + f"{c['n'][c['bin'] == 'top'].max():,}"
                  + " most active)", fontsize=8)
    fig.supylabel(f"median {MEASUREMENTS[column][0]}", fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.08, top=0.96, wspace=0.12, hspace=0.38)
    return fig


def named_figure(named: list[str], curves: pd.DataFrame, spearman: pd.DataFrame,
                 medians: pd.DataFrame, n_bins: int) -> plt.Figure:
    """Per named concept a 2 x 4 block, one panel per measurement."""
    columns = list(MEASUREMENTS)
    side = FIG_W_CM / 2.54
    fig, axes = plt.subplots(2 * len(named), 4, figsize=(side, side * 0.5 * len(named)), squeeze=False)
    for i, letter in enumerate(named):
        for k, column in enumerate(columns):
            ax = axes[2 * i + k // 4, k % 4]
            c, s, m = (d[d["measurement"] == column] for d in (curves, spearman, medians))
            draw_panel(ax, c[c["concept"] == letter], m[m["concept"] == letter].iloc[0], n_bins)
            ax.set_ylim(*y_range(c[c["concept"].isin(named)], m[m["concept"].isin(named)]))
            r = s[s["concept"] == letter].iloc[0]
            ax.set_title(rf"{letter}, {MEASUREMENTS[column][0]}" + "\n"
                         + rf"$\rho = {r['rho']:.2f}\ [{r['ci_low']:.2f},\ {r['ci_high']:.2f}]$",
                         fontsize=6.5, pad=2)
    fig.supxlabel(r"activation decile of $\kappa_j$", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.09, top=0.92, wspace=0.35, hspace=0.75)
    return fig


def summary_table(spearman: pd.DataFrame, medians: pd.DataFrame, letters: list[str],
                  sections: dict[str, str], size_min_rho: float) -> pd.DataFrame:
    table = medians.merge(spearman, on=["concept", "measurement"])
    rows = []
    for letter in letters:
        t = table[table["concept"] == letter].set_index("measurement")
        row = {"concept": letter, "section": sections[letter], "frequency": t["frequency"].iloc[0]}
        for column in MEASUREMENTS:
            for field in ("median_active", "median_all", "rho", "ci_low", "ci_high"):
                row[f"{column}_{field}"] = t.loc[column, field]
        dominant = t["rho"].abs().idxmax()
        row |= {"dominant_measurement": dominant, "dominant_rho": t.loc[dominant, "rho"],
                "tracks_size": abs(t.loc["area_um2", "rho"]) >= size_min_rho}
        rows.append(row)
    return pd.DataFrame(rows)


def latex_tables(summary: pd.DataFrame) -> str:
    cols = list(MEASUREMENTS)

    def rho(r: pd.Series, c: str) -> str:
        return f"${r[f'{c}_rho']:.2f}_{{[{r[f'{c}_ci_low']:.2f},\\,{r[f'{c}_ci_high']:.2f}]}}$"

    def med(r: pd.Series, c: str) -> str:
        digits = 1 if r[f"{c}_median_all"] >= 10 else 2
        return f"{r[f'{c}_median_active']:.{digits}f} / {r[f'{c}_median_all']:.{digits}f}"

    out = []
    for title, cell, extra in (("Spearman $\\rho$ [95\\,\\% CI] of $\\kappa_j$ with each measurement (active nuclei)", rho, True),
                               ("Median of each measurement: active nuclei / all nuclei", med, False)):
        head = "Concept & Freq.\\ (\\%) & " + " & ".join(SHORT[c] for c in cols) + (" & Size" if extra else "")
        lines = [f"% {title}", "\\begin{tabular}{l r " + "r " * len(cols) + ("c" if extra else "") + "}",
                 "\\toprule", head + " \\\\", "\\midrule"]
        for _, r in summary.iterrows():
            cells = [r["concept"], f"{100 * r['frequency']:.1f}", *(cell(r, c) for c in cols)]
            if extra:
                cells.append("\\checkmark" if r["tracks_size"] else "")
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        out.append("\n".join(lines))
    return "\n".join(out)


def write_readme(path: Path, config: DictConfig, features: dict, summary: pd.DataFrame,
                 consistency: dict | None, commit: str, dirty: bool) -> None:
    size = summary.loc[summary["tracks_size"], "concept"].tolist()
    lines = [
        "# Concept morphometry (SAE k = 16, seed 1)",
        "",
        "How classic nuclear morphometry changes with the activation kappa_j of each concept.",
        (f"Code: `xai/morphometry_features.py` (step 1), `xai/concept_morphometry.py` (steps 2-3), "
        f"commit `{commit}`{' (the repository had other uncommitted changes)' if dirty else ''}."),
        "",
        "## Data",
        "",
        (f"- Test: {features['test']['n_nuclei']:,} nuclei of {features['test']['n_slides']} slides, the stored "
        "codes `codes_test.npy` (rows = `activations/test/index.parquet`)."),
        (f"- SAE `{Path(config.sae_model).name}` (k = 16, seed 1, lambda = 3), fit on validation; the concepts are "
        "fixed, so this describes them (no selection on test)."),
        ("- Polygons: the LSP-DETR segmentation (64 vertices, level-0 px; per slide `slide_nuclei_path`, sorted by "
        "`id`, checked against the index's id checksum and nucleus ids), converted to um with the slide's mpp "
        "(0.4862 um/px for all slides)."),
        "- Concepts are named by their letter in the review deck only (the letter -> index key stays local).",
        "",
        "## Measurements",
        "",
        ("- area (shoelace), perimeter, long / short axis = 2 x the semi-axes of the polygon's moment ellipse "
        "(4 sqrt of the eigenvalues of its area covariance), axis ratio = long / short, circularity = "
        "4 pi area / perimeter^2: all nuclei."),
        ("- hematoxylin OD: level 0 (0.486 um/px), `skimage.color.rgb2hed` (Ruifrok-Johnston), mean of the "
        "hematoxylin channel over the pixels whose centre lies in the polygon (pixel count / area = 1.00); "
        "integrated OD = mean x area. Normalized per slide by the median mean OD of the slide's random-sample "
        "nuclei (the top-activated nuclei are left out of the reference)."),
        (f"- OD subsample (seed {config.seed}): {features['test']['od']['max_per_slide']} random nuclei per slide "
        f"({features['test']['n_od_random']:,}) plus each concept's {features['test']['od']['top_per_concept']:,} "
        f"most active test nuclei ({features['test']['n_od_top_only']:,} not already in the random sample); "
        f"{features['test']['n_od_measured']:,} measured."),
        "",
        "## Curves and correlation",
        "",
        (f"- Active nuclei (kappa_j > 0) in {config.n_bins} quantile bins of kappa_j (edges from all active nuclei, "
        "shared by all measurements); per bin the median and quartiles; plus a `top` point, the concept's "
        f"{config.top_n:,} most active nuclei. OD statistics use the random sample only (the top point: all "
        "OD-measured top nuclei). The random sample takes up to 500 nuclei per slide, so OD statistics weight "
        "slides more evenly than the polygon statistics (all nuclei)."),
        "- References: median over all nuclei (dashed) and over the inactive nuclei, kappa_j = 0 (dotted).",
        (f"- Spearman rho over the active nuclei (OD: of the random sample); 95 % percentile CI from "
        f"{config.n_bootstrap:,} resamples of slides (seed {config.seed}; ranks of the full sample kept, "
        "Pearson of the ranks over the resampled slides). Effect sizes only: with millions of nuclei p-values are "
        "meaningless."),
        (f"- `tracks_size`: |rho| with the area >= {config.size_min_rho:g} (the axes alone can also mean "
        "elongation: a falling short axis with a rising axis ratio). Size enters the model directly (log s, the first-harmonic semi-major axis, is an input "
        "feature, as is the orientation psi), so size-related concepts are expected. "
        f"Concepts that mainly track size: {', '.join(size) if size else 'none'}."),
        "",
    ]
    if consistency is not None:
        lines += [
            "## Validation (consistency check)",
            "",
            (f"Polygon features on validation ({features['validation']['n_nuclei']:,} nuclei, "
            f"{features['validation']['n_slides']} slides), codes from the same SAE (forward pass, CPU fp32; "
            "it reproduces the stored TF32 test codes to 2e-3). rho test vs. validation over all "
            f"{consistency['n_pairs']} concept x measurement pairs: max |difference| {consistency['max_abs_diff']:.3f}, "
            f"median {consistency['median_abs_diff']:.3f}; sign agreement {consistency['sign_agreement']:.0%} "
            f"(of pairs with |rho| >= 0.1: {consistency['sign_agreement_large']:.0%})."),
            "",
        ]
    lines += [
        "## Files",
        "",
        "- `morphometry_test.parquet`, `morphometry_validation.parquet`: per nucleus (not logged to MLflow; size).",
        "- `codes_validation.npy`: validation codes (local only).",
        "- `activation_curves.csv`, `spearman.csv`, `summary_table.csv`, `summary_table.tex` (booktabs).",
        "- `activation_curves_validation.csv`, `spearman_validation.csv`, `spearman_test_vs_validation.csv`.",
        ("- `figures/curves_<measurement>.pdf/png`, `figures/curves_named_concepts.pdf/png`, "
        "`figures/rho_test_vs_validation.png`."),
        "",
        "## Versions",
        "",
        (f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scipy "
        f"{scipy.__version__}, scikit-image {skimage.__version__}, matplotlib {mpl.__version__}."),
    ]
    path.write_text("\n".join(lines) + "\n")


def load_split(config: DictConfig, split: str, columns: list[str]) -> tuple[NDArray, pd.DataFrame, NDArray]:
    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == split].sort_values("offset")
    out = Path(config.morphometry_dir)
    codes_path = Path(config.concept_dir) / "codes_test.npy" if split == "test" else out / f"codes_{split}.npy"
    codes = np.load(codes_path, mmap_mode="r")
    morphometry = pd.read_parquet(out / f"morphometry_{split}.parquet", columns=columns)
    slide = np.repeat(np.arange(len(slides)), slides["n_nuclei"].to_numpy())
    assert len(codes) == len(morphometry) == len(slide)
    return codes, morphometry, slide


@with_cli_args(["+xai=concept_morphometry"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    out = Path(config.morphometry_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    concepts = concept_letters(config.deck_key)
    letters = [letter for _, letter in concepts]
    sections = pd.read_csv(config.deck_key).set_index("letter")["section"].to_dict()
    features = {split: json.loads((out / f"morphometry_features_{split}.json").read_text())
                for split in ("test", "validation")}

    codes, morphometry, slide = load_split(config, "test", [*MEASUREMENTS, "od_random_sample"])
    curves, spearman, medians = analyse(codes, morphometry, slide, concepts, list(MEASUREMENTS), config)
    curves.to_csv(out / "activation_curves.csv", index=False)
    spearman.to_csv(out / "spearman.csv", index=False)
    summary = summary_table(spearman, medians, letters, sections, config.size_min_rho)
    summary.to_csv(out / "summary_table.csv", index=False)
    (out / "summary_table.tex").write_text(latex_tables(summary))

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

    consistency = None
    if config.validation:
        polygon = [c for c, (_, od, _) in MEASUREMENTS.items() if not od]
        codes_v, morphometry_v, slide_v = load_split(config, "validation", polygon)
        curves_v, spearman_v, _ = analyse(codes_v, morphometry_v, slide_v, concepts, polygon, config)
        curves_v.to_csv(out / "activation_curves_validation.csv", index=False)
        spearman_v.to_csv(out / "spearman_validation.csv", index=False)
        both = spearman.merge(spearman_v, on=["concept", "measurement"], suffixes=("_test", "_validation"))
        both.to_csv(out / "spearman_test_vs_validation.csv", index=False)
        diff = (both["rho_test"] - both["rho_validation"]).abs()
        large = both["rho_test"].abs() >= 0.1
        consistency = {"n_pairs": len(both), "max_abs_diff": float(diff.max()),
                       "median_abs_diff": float(diff.median()),
                       "sign_agreement": float((np.sign(both["rho_test"]) == np.sign(both["rho_validation"])).mean()),
                       "sign_agreement_large": float((np.sign(both["rho_test"][large])
                                                      == np.sign(both["rho_validation"][large])).mean())}
        with mpl.rc_context(THESIS_RC):
            fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))
            ax.plot([-1, 1], [-1, 1], color="#b4b4b4", lw=0.6)
            ax.scatter(both["rho_test"], both["rho_validation"], s=6, color=LINE_COLOR)
            ax.set_xlabel(r"$\rho$ (test)")
            ax.set_ylabel(r"$\rho$ (validation)")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out / "figures" / "rho_test_vs_validation.png", dpi=200)
            plt.close(fig)

    commit, dirty = code_commit()
    write_readme(out / "README.md", config, features, summary, consistency, commit, dirty)
    logger.log_hyperparams({"code_commit": commit, "code_dirty": dirty, "n_bins": config.n_bins,
                            "top_n": config.top_n, "n_bootstrap": config.n_bootstrap, "seed": config.seed,
                            "sae_model": Path(config.sae_model).name, "named_concepts": ",".join(config.named_concepts),
                            **{f"link/{k}": v for k, v in config.links.items()}})
    for name in ["README.md", "activation_curves.csv", "spearman.csv", "summary_table.csv", "summary_table.tex",
                 "morphometry_features_test.json", "morphometry_features_validation.json", "od_reference_test.csv"]:
        logger.log_artifact(str(out / name))
    if config.validation:
        for name in ["activation_curves_validation.csv", "spearman_validation.csv", "spearman_test_vs_validation.csv"]:
            logger.log_artifact(str(out / name))
    logger.log_artifacts(str(out / "figures"), "figures")
    for script in ("xai/morphometry_features.py", "xai/concept_morphometry.py"):
        logger.log_artifact(str(Path(__file__).parent.parent / script), "code")


if __name__ == "__main__":
    main()
