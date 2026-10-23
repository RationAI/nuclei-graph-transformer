"""Comparison of the TopK SAEs with the 16 concepts of the k = 16 SAE (step 4).

Concepts of the k = 16 SAE are those of the analysed seed (`reference_seed`), named by
their review letter and SAE index (`deck_key`). φ = z · wᵀu with u in the units of h
(TopK: c · W_dec column), on all test nuclei unless stated otherwise.

Undercomplete TopK (m = 16, K = 4) against the k = 16 SAE:
- `undercomplete_matching.csv`: Hungarian matching of unit-norm decoder directions
  (signed cosine, as `xai.evaluate_sae.stability`) of every m = 16 seed with every
  k = 16 seed; matched latent and cosine per k = 16 concept;
- `undercomplete_stability.csv`: per seed and latent the matched cosine to each other
  seed; stable at threshold t if ≥ t to both (0.8 is the k = 16 criterion), for both
  the m = 16 TopK and the k = 16 SAE;
- `undercomplete_shares.csv`: share of mean |φ| per concept of each m = 16 seed next to
  its Hungarian match in the k = 16 SAE (seed 1).

Main configuration (m = 1024, K = 16), every seed:
- `importance.csv`: per latent mean |φ|, its share, rank, frequency and wᵀu; the
  number of latents carrying 50 / 80 / 90 % of the total mean |φ| in
  `concentration.csv` (with the k = 16 SAE);
- `matching.csv`, concept -> latents: the latent with the highest decoder cosine
  with u_j, the one with the highest Pearson correlation of activations (κ_j and z_l
  on `subsample` fixed random test nuclei), and the latents contained in the concept,
  P(κ_j > 0 | z_l > 0) ≥ `containment` (latents active on ≥ `min_active` subsample
  nuclei); the Pearson correlation of their summed φ with φ_j on the subsample.
  Category: `match` (cosine > `match_cosine` or correlation > `match_correlation`),
  else `split` (contained latents), else `none`;
- `matching_latents.csv`, latents -> concepts: for the `top_latents` latents by mean
  |φ| the best concept by cosine, by correlation and by containment; `new_pattern`
  if cosine < `new_cosine` and correlation < `new_correlation` for every concept;
- `undercomplete_stable_eight.csv`: the matched latent and cosine of each of the
  stable eight concepts (seed 1) for every m = 16 seed, with mean and minimum;
- `morphometry/<model>/`: activation curves, Spearman rho with slide bootstrap and
  medians of the top latents in the format of `xai.concept_morphometry` (polygon
  features of all test nuclei; hematoxylin OD only where measured: the random
  per-slide sample, so the `top` point uses the random-sample members of the latent's
  `top_n` most active nuclei; their number is in `od_top_n.csv`, and points with fewer
  than `min_od_top_n` are marked `reliable = False` and left blank in the figures), and
  the validation check (polygon features).
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
import scipy.sparse
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import LinearHead, concept_logit_weights
from nuclei_graph.xai.runtime import setup_precision
from nuclei_graph.xai.sae import load_autoencoder
from nuclei_graph.xai.stability import cosine_similarity, matched_cosines
from nuclei_graph.xai.topk import encode_topk, load_topk
from xai.concept_masks import code_commit
from xai.concept_morphometry import (
    MEASUREMENTS,
    THESIS_RC,
    bin_curve,
    draw_panel,
    spearman_ci,
    summary_row,
    y_range,
)


SHARES = (0.5, 0.8, 0.9)


def unit_rows(weight: torch.Tensor) -> NDArray[np.float64]:
    w = weight.detach().double().numpy()
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def concentration(mean_abs_phi: NDArray) -> dict[str, int]:
    """Number of latents carrying 50 / 80 / 90 % of the total mean |φ|."""
    cumulative = np.cumsum(np.sort(mean_abs_phi)[::-1]) / mean_abs_phi.sum()
    return {
        f"latents_{int(q * 100)}pct": int(np.searchsorted(cumulative, q) + 1)
        for q in SHARES
    }


def sparse_importance(
    indices: NDArray, values: NDArray, logit_weights: NDArray
) -> pd.DataFrame:
    """Per latent: mean z, frequency, wᵀu, mean |φ| = |wᵀu| · mean z (z ≥ 0)."""
    m, n = len(logit_weights), len(indices)
    flat_i, flat_v = indices.ravel().astype(np.int64), values.ravel().astype(np.float64)
    mean_z = np.bincount(flat_i, weights=flat_v, minlength=m) / n
    frequency = np.bincount(flat_i[flat_v > 0], minlength=m) / n
    mean_abs_phi = np.abs(logit_weights) * mean_z
    df = pd.DataFrame(
        {
            "latent": np.arange(m),
            "logit_weight": logit_weights,
            "sign": np.sign(logit_weights).astype(int),
            "frequency": frequency,
            "mean_z": mean_z,
            "mean_abs_phi": mean_abs_phi,
            "share_abs_phi": mean_abs_phi / mean_abs_phi.sum(),
        }
    )
    df["rank"] = df["mean_abs_phi"].rank(ascending=False, method="first").astype(int)
    return df


def cross_seed_stability(
    directions: dict[int, NDArray], thresholds: list[float], model: str
) -> pd.DataFrame:
    rows = []
    for a, reference in directions.items():
        cosines = {}
        for b, other in directions.items():
            if b != a:
                cosines[b] = matched_cosines(reference, other)[1]
        for j in range(len(reference)):
            row = {"model": model, "seed": a, "latent": j}
            row |= {f"cosine_seed{b}": float(c[j]) for b, c in cosines.items()}
            worst = min(c[j] for c in cosines.values())
            row |= {f"stable_{t:g}": bool(worst >= t) for t in thresholds}
            rows.append(row)
    return pd.DataFrame(rows)


def subsample_matrix(
    indices: NDArray, values: NDArray, rows: NDArray, m: int
) -> scipy.sparse.csr_matrix:
    idx, val = indices[rows].astype(np.int64), values[rows].astype(np.float64)
    n, k = idx.shape
    return scipy.sparse.csr_matrix(
        (val.ravel(), idx.ravel(), np.arange(0, n * k + 1, k)), shape=(n, m)
    )


def correlation(z: scipy.sparse.csr_matrix, kappa: NDArray) -> NDArray[np.float64]:
    """Pearson correlation of every column of z with every column of kappa, (m, k)."""
    n = z.shape[0]
    mean_z = np.asarray(z.mean(0)).ravel()
    std_z = np.sqrt(np.asarray(z.multiply(z).mean(0)).ravel() - mean_z**2)
    mean_k, std_k = kappa.mean(0), kappa.std(0)
    cov = (z.T @ kappa) / n - np.outer(mean_z, mean_k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.outer(std_z, std_k)


def containment_matrix(
    z: scipy.sparse.csr_matrix, kappa: NDArray, min_active: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """P(κ_j > 0 | z_l > 0), (m, k), NaN for latents active on < `min_active` nuclei."""
    active = z.copy()
    active.data = (active.data > 0).astype(np.float64)
    active.eliminate_zeros()
    counts = active.T @ (kappa > 0).astype(np.float64)
    n_active = np.asarray(active.sum(0)).ravel().astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = counts / n_active[:, None]
    share[n_active < min_active] = np.nan
    return share, n_active


def match_concepts(
    seed: int,
    concepts: pd.DataFrame,
    cosine: NDArray,
    corr: NDArray,
    contained: NDArray,
    z: scipy.sparse.csr_matrix,
    phi_concepts: NDArray,
    importance: pd.DataFrame,
    config: DictConfig,
) -> list[dict]:
    logit_weights = importance["logit_weight"].to_numpy()
    rows = []
    for c in concepts.itertuples(index=False):
        j = c.concept
        best_cos, best_corr = (
            int(np.nanargmax(cosine[:, j])),
            int(np.nanargmax(corr[:, j])),
        )
        group = np.flatnonzero(contained[:, j] >= config.containment)
        group = group[np.argsort(importance["rank"].to_numpy()[group])]
        group_corr = np.nan
        if len(group):
            phi_group = z[:, group] @ logit_weights[group]
            group_corr = float(np.corrcoef(phi_group, phi_concepts[:, j])[0, 1])
        if (
            cosine[best_cos, j] > config.match_cosine
            or corr[best_corr, j] > config.match_correlation
        ):
            category = "match"
        elif len(group):
            category = "split"
        else:
            category = "none"
        rows.append(
            {
                "model": f"topk_m{config.main.num_latents}_k{config.main.k}_seed{seed}",
                "seed": seed,
                "letter": c.letter,
                "sae_index": j,
                "stable_k16": c.stable,
                "k16_share_abs_phi": c.share_abs_phi,
                "best_cosine_latent": best_cos,
                "best_cosine": float(cosine[best_cos, j]),
                "best_cosine_latent_correlation": float(corr[best_cos, j]),
                "best_correlation_latent": best_corr,
                "best_correlation": float(corr[best_corr, j]),
                "best_correlation_latent_cosine": float(cosine[best_corr, j]),
                "n_contained": len(group),
                "contained_latents_by_importance": ";".join(map(str, group[:10])),
                "contained_share_abs_phi": float(
                    importance["share_abs_phi"].to_numpy()[group].sum()
                ),
                "contained_summed_phi_correlation": group_corr,
                "category": category,
            }
        )
    return rows


def match_latents(
    seed: int,
    concepts: pd.DataFrame,
    cosine: NDArray,
    corr: NDArray,
    contained: NDArray,
    n_active: NDArray,
    importance: pd.DataFrame,
    config: DictConfig,
) -> list[dict]:
    letters = concepts.set_index("concept")["letter"]
    top = importance.nsmallest(config.top_latents, "rank")
    rows = []
    for latent in top.itertuples(index=False):
        l = latent.latent
        j_cos, j_corr = int(np.nanargmax(cosine[l])), int(np.nanargmax(corr[l]))
        has_containment = np.isfinite(contained[l]).any()
        j_cont = int(np.nanargmax(contained[l])) if has_containment else -1
        rows.append(
            {
                "model": f"topk_m{config.main.num_latents}_k{config.main.k}_seed{seed}",
                "seed": seed,
                "rank": latent.rank,
                "latent": l,
                "share_abs_phi": latent.share_abs_phi,
                "frequency": latent.frequency,
                "logit_weight": latent.logit_weight,
                "best_cosine_letter": letters[j_cos],
                "best_cosine_sae_index": j_cos,
                "best_cosine": float(cosine[l, j_cos]),
                "best_correlation_letter": letters[j_corr],
                "best_correlation_sae_index": j_corr,
                "best_correlation": float(corr[l, j_corr]),
                "max_containment_letter": letters[j_cont] if has_containment else "",
                "max_containment": float(contained[l, j_cont])
                if has_containment
                else np.nan,
                "active_subsample": int(n_active[l]),
                "match": bool(
                    cosine[l, j_cos] > config.match_cosine
                    or corr[l, j_corr] > config.match_correlation
                ),
                "new_pattern": bool(
                    cosine[l, j_cos] < config.new_cosine
                    and corr[l, j_corr] < config.new_correlation
                ),
            }
        )
    return rows


def latent_morphometry(
    codes: NDArray,
    names: list[str],
    morphometry: pd.DataFrame,
    slide: NDArray,
    measurements: list[str],
    config: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`xai.concept_morphometry.analyse` for latents; OD only where measured."""
    m = config.morphometry
    rng = np.random.default_rng(config.seed)
    n_slides = int(slide.max()) + 1
    random = (
        morphometry["od_random_sample"].to_numpy()
        if "od_random_sample" in morphometry
        else None
    )
    curves, spearman, medians = [], [], []
    for j, name in enumerate(names):
        kappa = np.asarray(codes[:, j], dtype=np.float64)
        active = kappa > 0
        edges = np.quantile(kappa[active], np.linspace(0, 1, m.n_bins + 1)[1:-1])
        bins = np.searchsorted(edges, kappa, side="right")
        top = np.argpartition(-kappa, m.top_n)[: m.top_n]
        top = top[kappa[top] > 0]
        for column in measurements:
            values = morphometry[column].to_numpy(dtype=np.float64)
            od = MEASUREMENTS[column][1]
            population = random if od else np.ones(len(kappa), dtype=bool)
            on, off = active & population, ~active & population
            top_measured = top[population[top]]
            assert np.isfinite(values[top_measured]).all(), (name, column)
            rows = [
                summary_row("inactive", kappa[off], values[off]),
                *bin_curve(kappa[on], values[on], bins[on], m.n_bins),
            ]
            if len(top_measured):
                rows.append(
                    summary_row("top", kappa[top_measured], values[top_measured])
                )
            else:
                rows.append(
                    {
                        "bin": "top",
                        "n": 0,
                        "median": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                    }
                )
            # OD of the top point: the random-sample members of the top nuclei only
            rows[-1]["reliable"] = not od or len(top_measured) >= m.min_od_top_n
            curves += [{"latent": name, "measurement": column, **r} for r in rows]
            spearman.append(
                {
                    "latent": name,
                    "measurement": column,
                    "population": "random sample" if od else "all",
                    **spearman_ci(
                        kappa[on], values[on], slide[on], n_slides, m.n_bootstrap, rng
                    ),
                }
            )
            medians.append(
                {
                    "latent": name,
                    "measurement": column,
                    "frequency": float(active.mean()),
                    "median_active": float(np.median(values[on])),
                    "median_inactive": float(np.median(values[off])),
                    "median_all": float(np.median(values[population])),
                }
            )
    return pd.DataFrame(curves), pd.DataFrame(spearman), pd.DataFrame(medians)


def grid_figure(
    column: str,
    curves: pd.DataFrame,
    spearman: pd.DataFrame,
    medians: pd.DataFrame,
    names: list[str],
    n_bins: int,
    title: str,
) -> plt.Figure:
    """As `xai.concept_morphometry.grid_figure`, for any number of latents (5 columns)."""
    c, s, m = (d[d["measurement"] == column] for d in (curves, spearman, medians))
    cols = 5
    rows = int(np.ceil(len(names) / cols))
    side = 16.0 / 2.54
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(side, side * 0.23 * rows + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, name in zip(axes.flat, names, strict=False):  # empty panels hidden below
        curve = c[c["latent"] == name].copy()
        unreliable = curve["reliable"].eq(
            False
        )  # OD top point with too few nuclei: blank
        curve.loc[unreliable, ["median", "q25", "q75"]] = np.nan
        draw_panel(ax, curve, m[m["latent"] == name].iloc[0], n_bins)
        rho = s[s["latent"] == name].iloc[0]["rho"]
        ax.set_title(rf"{name}: $\rho = {rho:.2f}$", fontsize=7, pad=2)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    axes[0, 0].set_ylim(*y_range(c.dropna(subset=["q25", "q75"]), m))
    fig.suptitle(title, fontsize=8)
    fig.supxlabel("activation decile of $z_l$ (top: most active)", fontsize=8)
    fig.supylabel(f"median {MEASUREMENTS[column][0]}", fontsize=8)
    height = fig.get_figheight()  # margins in inches, for any number of rows
    fig.subplots_adjust(
        left=0.1,
        right=0.99,
        bottom=0.5 / height,
        top=1 - 0.55 / height,
        wspace=0.12,
        hspace=0.45,
    )
    return fig


def morphometry_split(
    config: DictConfig, split: str, columns: list[str]
) -> tuple[pd.DataFrame, NDArray]:
    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == split].sort_values("offset")
    path = Path(config.morphometry.features_dir) / f"morphometry_{split}.parquet"
    morphometry = pd.read_parquet(path, columns=columns)
    slide = np.repeat(np.arange(len(slides)), slides["n_nuclei"].to_numpy())
    assert len(morphometry) == len(slide)
    return morphometry, slide


def dense_columns(indices: NDArray, values: NDArray, latents: list[int]) -> NDArray:
    out = np.zeros((len(indices), len(latents)), dtype=np.float32)
    for j, latent in enumerate(latents):
        rows, slots = np.nonzero(indices == latent)
        out[rows, j] = values[rows, slots]
    return out


@with_cli_args(["+xai=compare_topk_concepts"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog(
    log_config=False, log_stream=False
)  # attached to the TopK run: keep its configs/ and console.log
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    for key, value in setup_precision(config.matmul_precision).items():
        client.set_tag(logger.run_id, key, value)
    commit, dirty = code_commit("xai", "nuclei_graph", "configs")
    client.set_tag(
        logger.run_id,
        "xai.comparison_code_commit",
        f"{commit}{' (dirty)' if dirty else ''}",
    )
    torch.set_num_threads(config.num_threads)
    out = Path(config.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    resolved = out / "compare_topk_concepts_config.yaml"
    resolved.write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(resolved), "configs_comparison")
    for source in (__file__, "configs/xai/compare_topk_concepts.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")

    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    test = load_activations(config.activations_dir, "test")
    key = pd.read_csv(config.deck_key)
    concepts = pd.DataFrame(
        {"concept": key["concept"].astype(int), "letter": key["letter"].astype(str)}
    )
    concepts["stable"] = concepts["concept"].isin(list(config.stable_concepts))
    stable_letters = concepts.loc[concepts["stable"], "letter"].tolist()
    assert (
        key.set_index("concept").loc[list(config.stable_concepts), "stability"]
        == "stable"
    ).all()

    # the k = 16 SAE: directions of every seed, codes and φ of the analysed seed
    k16 = {s: load_autoencoder(p)[0] for s, p in config.reference_models.items()}
    k16_dirs = {s: unit_rows(model.decoder_weight) for s, model in k16.items()}
    reference = k16[config.reference_seed]
    U_ref = reference.decoder_weight.detach().double().numpy()
    w_ref = concept_logit_weights(U_ref, head)
    kappa = np.load(config.reference_codes, mmap_mode="r")
    assert kappa.shape == (len(test), 16)
    k16_mean_abs_phi = np.abs(w_ref) * np.asarray(kappa.mean(0, dtype=np.float64))
    k16_frequency = np.asarray((kappa > 0).mean(0))
    concepts["share_abs_phi"] = (k16_mean_abs_phi / k16_mean_abs_phi.sum())[
        concepts["concept"]
    ]
    deviation = float(
        np.abs(
            concepts.set_index("letter")["share_abs_phi"]
            - key.set_index("letter")["share_abs_phi"]
        ).max()
    )
    print(f"k = 16 |φ| shares vs. the review key: max deviation {deviation:.2e}")
    assert deviation < 1e-6
    concepts = concepts.sort_values("concept").reset_index(drop=True)

    # --- undercomplete TopK (m = 16, K = 4) vs. the k = 16 SAE ---
    uc = config.undercomplete
    uc_models = {
        s: load_topk(
            str(
                Path(config.topk_dir)
                / "models"
                / f"topk_m{uc.num_latents}_k{uc.k}_seed{s}.pt"
            )
        )[0]
        for s in uc.seeds
    }
    uc_dirs = {s: unit_rows(model.w_dec) for s, model in uc_models.items()}
    letter_of = concepts.set_index("concept")["letter"]
    rows = []
    for b, reference_dirs in k16_dirs.items():
        for a, dirs in uc_dirs.items():
            match, cosine = matched_cosines(reference_dirs, dirs)
            for j in range(len(reference_dirs)):
                is_ref = b == config.reference_seed
                rows.append(
                    {
                        "k16_seed": b,
                        "k16_concept": j,
                        "letter": letter_of[j] if is_ref else "",
                        "stable_k16": bool(is_ref and j in config.stable_concepts),
                        "m16_seed": a,
                        "m16_latent": int(match[j]),
                        "cosine": float(cosine[j]),
                    }
                )
    uc_matching = pd.DataFrame(rows)
    uc_matching.to_csv(out / "undercomplete_matching.csv", index=False)
    # the decision table: the stable eight of the analysed k = 16 seed, every m = 16 seed
    stable_eight = uc_matching[
        (uc_matching["k16_seed"] == config.reference_seed) & uc_matching["stable_k16"]
    ].pivot(
        index=["letter", "k16_concept"],
        columns="m16_seed",
        values=["cosine", "m16_latent"],
    )
    stable_eight.columns = [f"{v}_m16_seed{seed}" for v, seed in stable_eight.columns]
    latent_columns = stable_eight.filter(like="m16_latent_").columns
    stable_eight[latent_columns] = stable_eight[latent_columns].astype(int)
    cosines = stable_eight.filter(like="cosine_")
    stable_eight["cosine_mean"] = cosines.mean(axis=1)
    stable_eight["cosine_min"] = cosines.min(axis=1)
    stable_eight = stable_eight.reset_index().rename(
        columns={"k16_concept": "sae_index"}
    )
    stable_eight = stable_eight.sort_values("letter")
    stable_eight.to_csv(out / "undercomplete_stable_eight.csv", index=False)
    print(stable_eight.round(3).to_string(index=False))

    stability = pd.concat(
        [
            cross_seed_stability(
                uc_dirs, list(config.stability_thresholds), "topk_m16_k4"
            ),
            cross_seed_stability(
                k16_dirs, list(config.stability_thresholds), "sae_l1_k16"
            ),
        ],
        ignore_index=True,
    )
    stability.to_csv(out / "undercomplete_stability.csv", index=False)
    stable_counts = stability.groupby(["model", "seed"])[
        [f"stable_{t:g}" for t in config.stability_thresholds]
    ].sum()
    print(stable_counts.to_string())
    k16_stable = stability[
        (stability["model"] == "sae_l1_k16")
        & (stability["seed"] == config.reference_seed)
    ]
    assert sorted(k16_stable.loc[k16_stable["stable_0.8"], "latent"]) == sorted(
        config.stable_concepts
    )

    share_rows = []
    for a, model in uc_models.items():
        indices, values = encode_topk(model, test.h)
        w_uc = concept_logit_weights(
            model.decoder_weight.detach().double().numpy(), head
        )
        imp = sparse_importance(indices, values, w_uc)
        match, cosine = matched_cosines(k16_dirs[config.reference_seed], uc_dirs[a])
        for j in range(16):
            l = int(match[j])
            share_rows.append(
                {
                    "m16_seed": a,
                    "letter": letter_of[j],
                    "sae_index": j,
                    "stable_k16": j in config.stable_concepts,
                    "k16_share_abs_phi": float(concepts.loc[j, "share_abs_phi"]),
                    "k16_logit_weight": float(w_ref[j]),
                    "m16_latent": l,
                    "cosine": float(cosine[j]),
                    "m16_share_abs_phi": float(imp.loc[l, "share_abs_phi"]),
                    "m16_logit_weight": float(imp.loc[l, "logit_weight"]),
                    "m16_frequency": float(imp.loc[l, "frequency"]),
                    "k16_frequency": float(k16_frequency[j]),
                }
            )
    shares = pd.DataFrame(share_rows)
    shares.to_csv(out / "undercomplete_shares.csv", index=False)

    # --- main configuration: importance and matching ---
    rng = np.random.default_rng(config.seed)
    sub = np.sort(rng.choice(len(test), config.subsample, replace=False))
    kappa_sub = np.asarray(kappa[sub], dtype=np.float64)
    phi_sub = kappa_sub * w_ref
    main_cfg = config.main
    importance_parts, concept_rows, latent_rows = [], [], []
    concentration_rows = [
        {
            "model": f"sae_l1_k16_seed{config.reference_seed}",
            **concentration(k16_mean_abs_phi),
        }
    ]
    top_latents: dict[int, list[int]] = {}
    main_codes = {}
    for s in main_cfg.seeds:
        name = f"topk_m{main_cfg.num_latents}_k{main_cfg.k}_seed{s}"
        model = load_topk(str(Path(config.topk_dir) / "models" / f"{name}.pt"))[0]
        codes_dir = Path(config.topk_dir) / "codes" / name
        indices = np.load(codes_dir / "codes_test_indices.npy", mmap_mode="r")
        values = np.load(codes_dir / "codes_test_values.npy", mmap_mode="r")
        assert len(indices) == len(test)
        main_codes[s] = (indices, values, model)
        U = model.decoder_weight.detach().double().numpy()
        imp = sparse_importance(
            np.asarray(indices), np.asarray(values), concept_logit_weights(U, head)
        )
        importance_parts.append(imp.assign(model=name, seed=s))
        concentration_rows.append(
            {"model": name, **concentration(imp["mean_abs_phi"].to_numpy())}
        )
        top_latents[s] = imp.nsmallest(config.top_latents, "rank")["latent"].tolist()

        z = subsample_matrix(indices, values, sub, main_cfg.num_latents)
        cosine = cosine_similarity(U, U_ref)
        corr = correlation(z, kappa_sub)
        contained, n_active = containment_matrix(z, kappa_sub, config.min_active)
        concept_rows += match_concepts(
            s, concepts, cosine, corr, contained, z, phi_sub, imp, config
        )
        latent_rows += match_latents(
            s, concepts, cosine, corr, contained, n_active, imp, config
        )

    importance = pd.concat(importance_parts, ignore_index=True)
    importance.to_csv(out / "importance.csv", index=False)
    pd.DataFrame(concentration_rows).to_csv(out / "concentration.csv", index=False)
    matching = pd.DataFrame(concept_rows)
    matching.to_csv(out / "matching.csv", index=False)
    matching_latents = pd.DataFrame(latent_rows)
    matching_latents.to_csv(out / "matching_latents.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(concentration_rows).to_string(index=False))
        print(
            matching[
                [
                    "seed",
                    "letter",
                    "sae_index",
                    "best_cosine",
                    "best_correlation",
                    "n_contained",
                    "contained_summed_phi_correlation",
                    "category",
                ]
            ].to_string(index=False)
        )
        print(
            matching_latents[
                [
                    "seed",
                    "rank",
                    "latent",
                    "share_abs_phi",
                    "best_cosine_letter",
                    "best_cosine",
                    "best_correlation_letter",
                    "best_correlation",
                    "new_pattern",
                ]
            ].to_string(index=False)
        )

    # --- morphometry of the top latents ---
    m = config.morphometry
    test_morph, test_slide = morphometry_split(
        config, "test", [*MEASUREMENTS, "od_random_sample"]
    )
    polygon = [c for c, (_, od, _) in MEASUREMENTS.items() if not od]
    if m.validation:
        validation = load_activations(config.activations_dir, "validation")
        val_morph, val_slide = morphometry_split(config, "validation", polygon)
    consistency = {}
    for s, (indices, values, model) in main_codes.items():
        name = f"topk_m{main_cfg.num_latents}_k{main_cfg.k}_seed{s}"
        morph_dir = out / "morphometry" / name
        (morph_dir / "figures").mkdir(parents=True, exist_ok=True)
        latents = top_latents[s]
        names = [f"L{l}" for l in latents]
        codes = dense_columns(np.asarray(indices), np.asarray(values), latents)
        curves, spearman, medians = latent_morphometry(
            codes, names, test_morph, test_slide, list(MEASUREMENTS), config
        )
        curves.to_csv(morph_dir / "activation_curves.csv", index=False)
        od_top = curves[
            (curves["bin"] == "top")
            & curves["measurement"].isin(
                [col for col, (_, od, _) in MEASUREMENTS.items() if od]
            )
        ][["latent", "measurement", "n", "reliable"]]
        od_top.to_csv(morph_dir / "od_top_n.csv", index=False)
        print(
            name,
            "OD top n:",
            od_top.drop_duplicates("latent").set_index("latent")["n"].to_dict(),
        )
        spearman.to_csv(morph_dir / "spearman.csv", index=False)
        medians.to_csv(morph_dir / "medians.csv", index=False)
        with mpl.rc_context(THESIS_RC):
            for column in MEASUREMENTS:
                fig = grid_figure(
                    column,
                    curves,
                    spearman,
                    medians,
                    names,
                    m.n_bins,
                    rf"{name.replace('_', ' ')}: top {len(names)} latents by mean $|\varphi|$",
                )
                fig.savefig(morph_dir / "figures" / f"curves_{column}.pdf")
                fig.savefig(morph_dir / "figures" / f"curves_{column}.png", dpi=200)
                plt.close(fig)
        if m.validation:
            val_indices, val_values = encode_topk(model, validation.h)
            val_codes = dense_columns(val_indices, val_values, latents)
            curves_v, spearman_v, _ = latent_morphometry(
                val_codes, names, val_morph, val_slide, polygon, config
            )
            curves_v.to_csv(morph_dir / "activation_curves_validation.csv", index=False)
            both = spearman.merge(
                spearman_v,
                on=["latent", "measurement"],
                suffixes=("_test", "_validation"),
            )
            both.to_csv(morph_dir / "spearman_test_vs_validation.csv", index=False)
            diff = (both["rho_test"] - both["rho_validation"]).abs()
            large = both["rho_test"].abs() >= 0.1
            consistency[name] = {
                "n_pairs": len(both),
                "max_abs_diff": float(diff.max()),
                "median_abs_diff": float(diff.median()),
                "sign_agreement_large": float(
                    (
                        np.sign(both["rho_test"][large])
                        == np.sign(both["rho_validation"][large])
                    ).mean()
                ),
            }
    (out / "morphometry" / "validation_consistency.json").write_text(
        json.dumps(consistency, indent=2)
    )
    print(json.dumps(consistency, indent=2))

    summary = {
        "k16_share_deviation_from_key": deviation,
        "stable_letters": stable_letters,
        "stable_counts": {
            f"{model}/seed{seed}": row.to_dict()
            for (model, seed), row in stable_counts.iterrows()
        },
    }
    (out / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=int)
    )
    for name in (
        "undercomplete_matching.csv",
        "undercomplete_stability.csv",
        "undercomplete_shares.csv",
        "importance.csv",
        "concentration.csv",
        "matching.csv",
        "matching_latents.csv",
        "comparison_summary.json",
    ):
        logger.log_artifact(str(out / name), "comparison")
    logger.log_artifacts(str(out / "morphometry"), "comparison/morphometry")
    for (model, seed), row in stable_counts.iterrows():
        logger.log_metrics(
            {f"comparison/{model}/seed{seed}/{k}": float(v) for k, v in row.items()}
        )


if __name__ == "__main__":
    main()
