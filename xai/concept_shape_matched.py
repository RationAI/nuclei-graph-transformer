"""Shape-matched control for the concepts' tissue context (no MLflow run).

For each concept in `violin_concepts`, each of its `top_n` most strongly activated test nuclei
is matched to the nearest inactive nucleus (kappa_j = 0) of the same slide, with replacement
(if the slide has no inactive nucleus: globally, see below), in two ways (`match_on`):

- `morphometry`: z-standardised area, long axis, short axis and circularity (means and SDs
  over all test nuclei), Euclidean; globally: among all inactive nuclei;
- `model_input`: the model's shape input (`NucleiFeatureExtractor.get_efd_features`, order 16:
  64 EFD coefficients + log s + cos 2psi + sin 2psi = 67), standardised with the means and SDs
  of a random sample (FIT_PER_SLIDE nuclei per slide; constant inputs, the normalised first
  harmonic, left out), Euclidean on the first principal components covering >= PCA_VARIANCE
  of the sample's variance; globally: among the inactive nuclei of that random sample.

For the top and the matched nuclei: the share in epithelium (benign + cancerous, mask >= 2),
the median 8-NN distance and the median signed boundary distance; differences top - matched
and top - all inactive with 95 % percentile intervals from `n_bootstrap` resamples of the top
nuclei's slides (a pair stays together; the median of all inactive nuclei is kept fixed).

Writes `context_shape_matched.csv` (with `match_on`), `context_shape_matched_balance.csv`
(per matching: SMD of the four morphometric features, top vs. matched and vs. all inactive,
and the largest |SMD| over the non-constant model inputs), `context_shape_matched_pca.json`
and `figures/context_shape_matched.pdf/png`. Concepts by letter only.
"""

import json
from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from scipy.spatial import cKDTree

from nuclei_graph.data.datasets.nuclei_features import NucleiFeatureExtractor
from xai.concept_morphometry import FIG_W_CM, THESIS_RC, concept_letters
from xai.prepare_nuclei import ids_checksum


MATCH_FEATURES = ["area_um2", "long_axis_um", "short_axis_um", "circularity"]
EFD_ORDER = 16  # as in training
FIT_PER_SLIDE = 300
PCA_VARIANCE = 0.95
N_JOBS = 8
# name -> (axis label, unit, kind)
OUTCOMES = {
    "epithelium": ("epithelium (benign + cancerous)", "pp", "share"),
    "knn8_mean_dist_um": ("8-NN distance", "um", "median"),
    "boundary_dist_um": ("boundary distance", "um", "median"),
}
REFERENCES = {  # column -> (legend label, colour)
    "inactive": ("all inactive nuclei", "#9a9a9a"),
    "morphometry": ("matched on area, axes, circularity", "#6b9bd1"),
    "model_input": ("matched on the model's shape input (PCA)", "#0b2f5e"),
}


def statistic(values: NDArray, kind: str) -> float:
    values = values[np.isfinite(values)]
    return float(100 * values.mean() if kind == "share" else np.median(values))


def slide_inputs(slide: dict) -> NDArray[np.float32]:
    """The model's 67-d shape input of all of the slide's nuclei, in index order."""
    nuclei = pd.read_parquet(slide["slide_nuclei_path"], columns=["id", "polygon"]).sort_values("id")
    assert ids_checksum(nuclei["id"].to_numpy()) == slide["ids_crc32"], slide["slide_id"]
    polygons = np.array(nuclei["polygon"].tolist(), dtype=np.float32).reshape(len(nuclei), -1, 2)
    extractor = NucleiFeatureExtractor()
    extractor.efd_order = EFD_ORDER
    return extractor.get_efd_features(polygons, float(slide["mpp_x"]), float(slide["mpp_y"]))


def fit_sample(slide: dict, seed: int) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """A random sample of the slide's nuclei: global rows and their 67-d inputs."""
    rng = np.random.default_rng([seed, int(slide["offset"])])
    inputs = slide_inputs(slide)
    local = np.sort(rng.choice(len(inputs), size=min(len(inputs), FIT_PER_SLIDE), replace=False))
    return local + int(slide["offset"]), inputs[local]


def match_slide(slide: dict, tops: dict[str, NDArray], inactive: dict[str, NDArray], mean: NDArray,
                scale: NDArray, keep: NDArray, components: NDArray, extra: dict[str, NDArray]) -> dict:
    """Model-input matches (global rows) of the slide's top nuclei per concept.

    Also the standardised inputs of the top, the matched and the `extra` (morphometry-matched) rows.
    """
    inputs = (slide_inputs(slide)[:, keep] - mean) / scale
    projected = inputs @ components
    offset = int(slide["offset"])
    matched, needed = {}, set()
    for letter, top in tops.items():
        pool = np.flatnonzero(inactive[letter])
        needed.update(top.tolist(), extra[letter].tolist())
        if len(pool) == 0:
            matched[letter] = np.full(len(top), -1)
            continue
        _, k = cKDTree(projected[pool]).query(projected[top - offset])
        matched[letter] = pool[k] + offset
        needed.update(matched[letter].tolist())
    rows = np.array(sorted(needed), dtype=np.int64)
    return {"matched": matched, "rows": rows, "inputs": inputs[rows - offset].astype(np.float32)}


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


def smd(a: NDArray, b: NDArray) -> NDArray:
    return (a.mean(axis=0) - b.mean(axis=0)) / np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2)


def figure(table: pd.DataFrame, letters: list[str], top_n: int) -> plt.Figure:
    side = FIG_W_CM / 2.54
    fig, axes = plt.subplots(len(OUTCOMES), 1, figsize=(side, side * 0.7), sharex=True)
    offsets = dict(zip(REFERENCES, (-0.2, 0.0, 0.2), strict=True))
    for ax, (name, (label, unit, _)) in zip(axes, OUTCOMES.items(), strict=True):
        t = table[table["outcome"] == name]
        for ref, (_, colour) in REFERENCES.items():
            if ref == "inactive":
                rows = t[t["match_on"] == "morphometry"].set_index("concept").loc[letters]
                diff, low, high = rows["diff_inactive"], rows["ci_low_inactive"], rows["ci_high_inactive"]
            else:
                rows = t[t["match_on"] == ref].set_index("concept").loc[letters]
                diff, low, high = rows["diff_matched"], rows["ci_low_matched"], rows["ci_high_matched"]
            x = np.arange(len(letters)) + offsets[ref]
            ax.errorbar(x, diff, yerr=[diff - low, high - diff], fmt="o", ms=3, lw=0.8, capsize=0, color=colour,
                        mec="none")
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(axis="y", color="#ececec", lw=0.4)
        ax.set_axisbelow(True)
        for i in range(1, len(letters)):
            ax.axvline(i - 0.5, color="#d9d9d9", lw=0.4, zorder=0)
        unit_label = "percentage points" if unit == "pp" else r"$\mu\mathrm{m}$"
        ax.set_title(f"{label[0].upper()}{label[1:]}: top {top_n:,} minus reference ({unit_label})", fontsize=8,
                     loc="left", pad=3)
        ax.tick_params(labelsize=7, length=2, width=0.4, pad=1.5)
        ax.tick_params(axis="x", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    axes[-1].set_xticks(range(len(letters)), letters, fontsize=8)
    axes[-1].set_xlim(-0.5, len(letters) - 0.5)
    handles = [mpl.lines.Line2D([], [], color=c, marker="o", ms=3, lw=0.8) for _, c in REFERENCES.values()]
    fig.legend(handles, [label for label, _ in REFERENCES.values()], loc="upper center", ncols=3, fontsize=7,
               frameon=False, borderaxespad=0.2)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.06, top=0.9, hspace=0.35)
    return fig


def morphometry_match(z: NDArray, top: NDArray, inactive: NDArray, slides: pd.DataFrame,
                      slide: NDArray) -> NDArray[np.int64]:
    """Nearest inactive nucleus on the standardised morphometric features.

    Of the same slide, otherwise (no inactive nucleus there) of all inactive nuclei.
    """
    matched = np.full(len(top), -1, dtype=np.int64)
    for s in np.unique(slide[top]):
        rows = np.flatnonzero(slide[top] == s)
        lo = int(slides.loc[s, "offset"])
        pool = lo + np.flatnonzero(inactive[lo: lo + int(slides.loc[s, "n_nuclei"])])
        if len(pool):
            _, k = cKDTree(z[pool]).query(z[top[rows]])
            matched[rows] = pool[k]
    if (matched < 0).any():
        pool = np.flatnonzero(inactive)
        _, k = cKDTree(z[pool]).query(z[top[matched < 0]])
        matched[matched < 0] = pool[k]
    return matched


@with_cli_args(["+xai=concept_morphometry"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
def main(config: DictConfig) -> None:
    out = Path(config.morphometry_dir)
    concepts = {letter: j for j, letter in concept_letters(config.deck_key)}
    letters = list(config.violin_concepts)
    codes = np.load(Path(config.concept_dir) / "codes_test.npy", mmap_mode="r")
    morphometry = pd.read_parquet(out / "morphometry_test.parquet", columns=MATCH_FEATURES)
    context = pd.read_parquet(out / "context_test.parquet",
                              columns=["compartment", "knn8_mean_dist_um", "boundary_dist_um"])
    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == "test"].sort_values("offset").reset_index(drop=True)
    slide = np.repeat(np.arange(len(slides)), slides["n_nuclei"].to_numpy())
    assert len(codes) == len(morphometry) == len(context) == len(slide)

    features = morphometry.to_numpy(dtype=np.float64)
    z = (features - features.mean(axis=0)) / features.std(axis=0)
    outcomes = {"epithelium": (context["compartment"].to_numpy() >= 2).astype(np.float64),
                "knn8_mean_dist_um": context["knn8_mean_dist_um"].to_numpy(dtype=np.float64),
                "boundary_dist_um": context["boundary_dist_um"].to_numpy(dtype=np.float64)}
    kappa = {letter: np.asarray(codes[:, concepts[letter]], dtype=np.float64) for letter in letters}
    tops = {}
    for letter in letters:
        top = np.argpartition(-kappa[letter], config.top_n)[: config.top_n]
        tops[letter] = np.sort(top[kappa[letter][top] > 0])

    # 1. morphometry match
    matched = {"morphometry": {letter: morphometry_match(z, tops[letter], kappa[letter] == 0, slides, slide)
                               for letter in letters}}

    # 2. model-input match: standardisation and PCA on a random sample, then per slide
    records = slides.to_dict("records")
    fitted = Parallel(n_jobs=N_JOBS)(delayed(fit_sample)(s, config.seed) for s in records)
    sample_rows = np.concatenate([rows for rows, _ in fitted])
    sample = np.concatenate([inputs for _, inputs in fitted])
    keep = sample.std(axis=0) > 1e-6
    mean, scale = sample[:, keep].mean(axis=0), sample[:, keep].std(axis=0)
    _, singular, vt = np.linalg.svd((sample[:, keep] - mean) / scale, full_matrices=False)
    explained = singular**2 / (singular**2).sum()
    n_components = int(np.searchsorted(np.cumsum(explained), PCA_VARIANCE) + 1)
    components = vt[:n_components].T
    pca = {"n_inputs": int(keep.size), "n_constant_inputs": int((~keep).sum()), "n_fit": len(sample),
           "n_components": n_components, "explained_variance": float(np.cumsum(explained)[n_components - 1])}

    jobs = []
    for s in np.unique(np.concatenate([slide[t] for t in tops.values()])):
        lo, hi = int(slides.loc[s, "offset"]), int(slides.loc[s, "offset"] + slides.loc[s, "n_nuclei"])
        in_slide = {letter: tops[letter][slide[tops[letter]] == s] for letter in letters}
        in_slide = {letter: t for letter, t in in_slide.items() if len(t)}
        extra = {letter: matched["morphometry"][letter][slide[tops[letter]] == s] for letter in in_slide}
        extra = {letter: e[(e >= lo) & (e < hi)] for letter, e in extra.items()}  # in this slide
        jobs.append((records[s], in_slide, {letter: kappa[letter][lo:hi] == 0 for letter in in_slide}, extra))
    results = Parallel(n_jobs=N_JOBS, verbose=2)(
        delayed(match_slide)(s, t, i, mean, scale, keep, components, e) for s, t, i, e in jobs)
    matched["model_input"] = {letter: np.full(len(tops[letter]), -1, dtype=np.int64) for letter in letters}
    for (record, in_slide, _, _), r in zip(jobs, results, strict=True):
        s = slide[int(record["offset"])]
        for letter in in_slide:
            matched["model_input"][letter][slide[tops[letter]] == s] = r["matched"][letter]
    input_rows = np.concatenate([r["rows"] for r in results])
    input_values = np.concatenate([r["inputs"] for r in results])
    input_of = {row: i for i, row in enumerate(input_rows.tolist())}

    # slides without an inactive nucleus: the nearest inactive nucleus of the random sample
    standardised_sample = ((sample[:, keep] - mean) / scale).astype(np.float32)
    for letter in letters:
        missing = matched["model_input"][letter] < 0
        if missing.any():
            pool = np.flatnonzero(kappa[letter][sample_rows] == 0)
            top_inputs = input_values[[input_of[r] for r in tops[letter][missing].tolist()]]
            _, k = cKDTree(standardised_sample[pool] @ components).query(top_inputs @ components)
            matched["model_input"][letter][missing] = sample_rows[pool[k]]
            for row, value in zip(sample_rows[pool[k]].tolist(), standardised_sample[pool[k]], strict=True):
                if row not in input_of:
                    input_of[row] = len(input_values)
                    input_values = np.vstack([input_values, value[None]])

    # inputs of morphometry matches from other slides (global fallback)
    elsewhere = sorted({r for m in matched["morphometry"].values() for r in m.tolist() if r not in input_of})
    for s in np.unique(slide[elsewhere]) if elsewhere else []:
        lo = int(slides.loc[s, "offset"])
        local = np.array([r for r in elsewhere if slide[r] == s]) - lo
        values = ((slide_inputs(records[s])[local][:, keep] - mean) / scale).astype(np.float32)
        for row, value in zip((local + lo).tolist(), values, strict=True):
            input_of[row] = len(input_values)
            input_values = np.vstack([input_values, value[None]])

    rng = np.random.default_rng(config.seed)
    rows, balance = [], []
    for letter in letters:
        top, inactive = tops[letter], kappa[letter] == 0
        for match_on, pairs in matched.items():
            ok = pairs[letter] >= 0
            t, m = top[ok], pairs[letter][ok]
            inputs_top = input_values[[input_of[r] for r in t.tolist()]]
            inputs_matched = input_values[[input_of[r] for r in m.tolist()]]
            balance.append({"concept": letter, "match_on": match_on, "n_unmatched": int((~ok).sum()),
                            "same_slide_share": float((slide[pairs[letter][ok]] == slide[t]).mean()),
                            **{f"smd_{name}": float(smd(features[t, k], features[m, k]))
                               for k, name in enumerate(MATCH_FEATURES)},
                            **{f"smd_{name}_vs_all_inactive": float(smd(features[t, k], features[inactive, k]))
                               for k, name in enumerate(MATCH_FEATURES)},
                            "max_abs_smd_model_inputs": float(np.abs(smd(inputs_top, inputs_matched)).max())})
            for name, (_, unit, kind) in OUTCOMES.items():
                values = outcomes[name]
                top_stat, matched_stat = statistic(values[t], kind), statistic(values[m], kind)
                inactive_stat = statistic(values[inactive], kind)
                rows.append({"concept": letter, "match_on": match_on, "outcome": name,
                             "unit": "%" if kind == "share" else unit, "n_top": len(t),
                             "n_slides": int(np.unique(slide[t]).size), "unique_controls": int(np.unique(m).size),
                             "top": top_stat, "matched": matched_stat, "inactive": inactive_stat,
                             "diff_matched": top_stat - matched_stat, "diff_inactive": top_stat - inactive_stat,
                             **bootstrap(values[t], values[m], slide[t], inactive_stat, kind, config.n_bootstrap,
                                         rng)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "context_shape_matched.csv", index=False)
    pd.DataFrame(balance).to_csv(out / "context_shape_matched_balance.csv", index=False)
    (out / "context_shape_matched_pca.json").write_text(json.dumps(pca, indent=2))
    with mpl.rc_context(THESIS_RC):
        fig = figure(table, letters, config.top_n)
        fig.savefig(out / "figures" / "context_shape_matched.pdf")
        fig.savefig(out / "figures" / "context_shape_matched.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
