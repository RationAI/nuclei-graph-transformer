"""Do the concepts reappear in independently trained models?

The l1 SAE of the thesis (k = 16, the analysed model's seed with the lowest holdout RE) is
trained on the representations of other training seeds of the same model (`xai.train_sae`,
`xai.evaluate_sae` in each model's own `out_dir`); this step matches their concepts to the
analysed model's stable concepts A-H.

The representation spaces of different models are not aligned, so decoder directions are
not compared across models. Concepts are matched by the Pearson correlation of their
activations κ on the same `subsample` fixed random test nuclei (seed `seed`, as
`xai.compare_topk_concepts`): Hungarian matching on the correlation (all 16 x 16) and,
per concept, its best partner. Per match: the sign of wᵀu_j in both models and the
correlation of their morphometric signatures (fig:res-signature: the `top_n` most active
test nuclei vs. the inactive ones, κ_j = 0; differences of medians, % for the nucleus and
neighbourhood measures, plain difference for orientation coherence, µm for the boundary
distance, percentage points for the stroma share). The attributes have different units,
so the signature correlation is the Pearson correlation of the 10 values after dividing
each attribute by its root mean square over all concepts of all models (`signature_r`);
the raw Pearson (`signature_r_raw`) and Spearman (`signature_rho`) are kept as well.
A concept is matched if its Hungarian partner has κ correlation ≥ `match.min_correlation`
and the same sign of wᵀu.

Per new model: the chosen seed (lowest holdout RE among the selected k = 16 SAEs, the rule
of the analysed model), its stable concepts (Hungarian decoder cosine ≥
`stability_threshold` to both other seeds of the same model), the faithfulness row of
tab:res-faithfulness (RE, explained variance and AUPRC of f_A = g(ẑ) on test, from
`completeness.csv`) and the contribution shares share_abs_phi = mean|φ_j| / Σ_k mean|φ_k|
on all test nuclei, φ_j = (wᵀu_j) c_j.

The per-nucleus morphometry and tissue context are properties of the nuclei, not of the
model, and are read from the analysed model's `concepts/k16_seed1/morphometry/` (same test
nuclei, same row order); the reference signature is recomputed and checked against its
`concept_signature.csv`.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import LinearHead, concept_logit_weights
from nuclei_graph.xai.runtime import setup_precision
from nuclei_graph.xai.sae import encode, load_autoencoder
from nuclei_graph.xai.stability import matched_cosines


# fig:res-signature: attribute -> unit ("%" relative difference of medians, "um" / "Δ"
# plain difference of medians, "pp" difference of shares)
SIGNATURE = {
    "area_um2": "%",
    "long_axis_um": "%",
    "short_axis_um": "%",
    "circularity": "%",
    "irregularity": "%",
    "knn8_mean_dist_um": "%",
    "pleomorphism_cv": "%",
    "orientation_coherence": "Δ",
    "boundary_dist_um": "um",
    "stroma": "pp",
}
MORPHOMETRY_COLUMNS = ["area_um2", "long_axis_um", "short_axis_um", "circularity"]
CONTEXT_COLUMNS = [
    "compartment",
    "irregularity",
    "knn8_mean_dist_um",
    "pleomorphism_cv",
    "orientation_coherence",
    "boundary_dist_um",
]
FAITHFULNESS = [
    "RE",
    "explained_variance_test",
    "FE",
    "AUPRC_fA",
    "AUPRC_f",
    "AUROC_fA",
    "AUROC_f",
    "L0",
]


# --- concepts of one model ---


def logit_weights(model, head: LinearHead) -> NDArray[np.float64]:
    """wᵀu_j for every concept."""
    return concept_logit_weights(model.decoder_weight.detach().double().numpy(), head)


def stable_concepts(
    sae_dir: Path, paths: dict[int, str], seed: int, threshold: float
) -> pd.DataFrame:
    """Per concept of `seed`: Hungarian decoder cosine to every other seed; stable if ≥ threshold in all."""
    reference = (
        load_autoencoder(str(sae_dir / paths[seed]))[0].decoder_weight.detach().numpy()
    )
    out = pd.DataFrame({"concept": np.arange(len(reference))})
    for other_seed, path in sorted(paths.items()):
        if other_seed == seed:
            continue
        other = load_autoencoder(str(sae_dir / path))[0].decoder_weight.detach().numpy()
        match, cosine = matched_cosines(reference, other)
        out[f"matched_concept_seed{other_seed}"] = match
        out[f"cosine_seed{other_seed}"] = cosine
    cosines = out.filter(like="cosine_seed")
    out["min_cosine"] = cosines.min(1)
    out["stable"] = (cosines >= threshold).all(1)
    return out


def contribution_shares(codes: NDArray, weights: NDArray) -> NDArray[np.float64]:
    """Share of mean|φ_j| over all nuclei, φ_j = (wᵀu_j) c_j."""
    mean_abs_c = np.zeros(codes.shape[1])
    for start in range(0, len(codes), 1 << 21):
        mean_abs_c += np.abs(codes[start : start + (1 << 21)].astype(np.float64)).sum(0)
    mean_abs_phi = np.abs(weights) * mean_abs_c / len(codes)
    return mean_abs_phi / mean_abs_phi.sum()


# --- signatures ---


def load_features(morphometry_dir: Path) -> dict[str, NDArray]:
    morphometry = pd.read_parquet(
        morphometry_dir / "morphometry_test.parquet", columns=MORPHOMETRY_COLUMNS
    )
    context = pd.read_parquet(
        morphometry_dir / "context_test.parquet", columns=CONTEXT_COLUMNS
    )
    values = {c: morphometry[c].to_numpy(np.float64) for c in MORPHOMETRY_COLUMNS}
    values |= {
        c: context[c].to_numpy(np.float64)
        for c in CONTEXT_COLUMNS
        if c != "compartment"
    }
    values["stroma"] = (context["compartment"].to_numpy() == 1).astype(np.float64)
    return values


def signature(
    kappa: NDArray, features: dict[str, NDArray], top_n: int
) -> dict[str, float]:
    """The top `top_n` active nuclei vs. the inactive ones (κ = 0), as fig:res-signature."""
    kappa = np.asarray(kappa, dtype=np.float64)
    order = np.argpartition(-kappa, top_n)[:top_n]
    top = np.zeros(len(kappa), dtype=bool)
    top[order[kappa[order] > 0]] = True
    inactive = kappa <= 0
    out = {}
    for name, unit in SIGNATURE.items():
        values = features[name]
        a, b = values[top], values[inactive]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if unit == "pp":
            out[name] = 100 * (a.mean() - b.mean())
        elif unit == "%":
            out[name] = 100 * (np.median(a) / np.median(b) - 1)
        else:
            out[name] = float(np.median(a) - np.median(b))
    return out


def signature_table(
    codes: NDArray, features: dict[str, NDArray], top_n: int
) -> pd.DataFrame:
    return pd.DataFrame(
        [signature(codes[:, j], features, top_n) for j in range(codes.shape[1])]
    )


def signature_correlations(a: NDArray, b: NDArray, scale: NDArray) -> dict[str, float]:
    return {
        "signature_r": float(np.corrcoef(a / scale, b / scale)[0, 1]),
        "signature_r_raw": float(np.corrcoef(a, b)[0, 1]),
        "signature_rho": float(spearmanr(a, b).statistic),
    }


# --- matching ---


def pearson_matrix(a: NDArray, b: NDArray) -> NDArray[np.float64]:
    """Pearson correlation between the columns of a (n, k) and b (n, m)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    return a.T @ b / len(a)


def match_concepts(
    reference_sub: NDArray,
    other_sub: NDArray,
    letters: dict[int, str],
    reference_w: NDArray,
    other_w: NDArray,
    reference_signature: pd.DataFrame,
    other_signature: pd.DataFrame,
    scale: NDArray,
    other_stable: NDArray[np.bool_],
    other_shares: NDArray,
    min_correlation: float,
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    corr = pearson_matrix(reference_sub, other_sub)
    rows_h, cols_h = linear_sum_assignment(-corr)
    hungarian = dict(zip(rows_h, cols_h, strict=True))
    rows = []
    for j, letter in letters.items():
        partner = int(hungarian[j])
        best = int(np.argmax(corr[j]))
        second = float(np.sort(corr[j])[-2])
        sig = signature_correlations(
            reference_signature.iloc[j].to_numpy(),
            other_signature.iloc[partner].to_numpy(),
            scale,
        )
        same_sign = bool(np.sign(reference_w[j]) == np.sign(other_w[partner]))
        rows.append(
            {
                "letter": letter,
                "concept": j,
                "partner": partner,
                "correlation": float(corr[j, partner]),
                "sign_reference": int(np.sign(reference_w[j])),
                "sign_partner": int(np.sign(other_w[partner])),
                "logit_weight_reference": float(reference_w[j]),
                "logit_weight_partner": float(other_w[partner]),
                "same_sign": same_sign,
                **sig,
                "partner_stable": bool(other_stable[partner]),
                "partner_share_abs_phi": float(other_shares[partner]),
                "best_partner": best,
                "best_correlation": float(corr[j, best]),
                "second_best_correlation": second,
                "best_is_hungarian": best == partner,
                "matched": bool(corr[j, partner] >= min_correlation and same_sign),
            }
        )
    return pd.DataFrame(rows), corr


# --- driver ---


def chosen_seed(models: pd.DataFrame, k: int) -> tuple[int, dict[int, str]]:
    sae = models[(models["method"] == "sae") & (models["k"] == k)]
    paths = dict(zip(sae["seed"].astype(int), sae["path"], strict=True))
    return int(sae.loc[sae["RE"].idxmin(), "seed"]), paths


def model_codes(
    model_dir: Path, path: str, device: str
) -> tuple[NDArray, NDArray, NDArray]:
    """Test codes (float32) of an SAE, its wᵀu and the test labels."""
    model, _ = load_autoencoder(str(model_dir / "sae" / path))
    head = LinearHead.from_npz(str(model_dir / "activations" / "head.npz"))
    test = load_activations(model_dir / "activations", "test")
    w = logit_weights(model, head)
    codes = encode(model.to(device), test.h, device=device).numpy()
    return codes, w, test


@with_cli_args(["+xai=cross_model_concepts"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    for key, value in setup_precision(config.matmul_precision).items():
        logger.experiment.set_tag(logger.run_id, key, value)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = config.num_concepts

    # reference: the analysed model
    ref_dir = Path(config.reference.dir)
    key = pd.read_csv(ref_dir / "session" / "key.csv")
    letters = {
        int(r.concept): r.letter for r in key[key["section"] == "main"].itertuples()
    }
    letters = dict(sorted(letters.items(), key=lambda item: item[1]))
    ref_model, ref_meta = load_autoencoder(str(ref_dir / "sae" / config.reference.sae))
    ref_w = logit_weights(
        ref_model, LinearHead.from_npz(str(ref_dir / "activations" / "head.npz"))
    )
    ref_codes = np.load(ref_dir / config.reference.codes)
    n_test = len(ref_codes)
    sub = np.sort(
        np.random.default_rng(config.seed).choice(
            n_test, config.subsample, replace=False
        )
    )
    ref_sub = np.asarray(ref_codes[sub], dtype=np.float64)

    features = load_features(ref_dir / config.reference.morphometry)
    signatures = {"reference": signature_table(ref_codes, features, config.top_n)}
    stored = pd.read_csv(
        ref_dir / config.reference.morphometry / "concept_signature.csv",
        index_col="concept",
    )
    recomputed = (
        signatures["reference"]
        .iloc[list(letters)]
        .set_index(pd.Index(list(letters.values())))
    )
    check = float((recomputed - stored[list(SIGNATURE)]).abs().max().max())
    print(f"reference signature vs. concept_signature.csv: max |Δ| = {check:.4f}")
    assert check < 0.01, (
        "The reference signature does not reproduce concept_signature.csv"
    )
    logger.log_metrics({"check/reference_signature_max_abs_diff": check})

    models, codes_w = {}, {}
    faithfulness, stability_rows = [], []
    for entry in config.models:
        name, model_dir = entry.name, Path(entry.dir)
        manifest = pd.read_csv(model_dir / "sae" / "models.csv")
        seed, paths = chosen_seed(manifest, k)
        assert len(paths) == 3, f"{name}: {len(paths)} selected seeds"
        print(f"{name}: seeds {sorted(paths)}, chosen seed {seed} ({paths[seed]})")
        stability = stable_concepts(
            model_dir / "sae", paths, seed, config.stability_threshold
        )

        codes, w, test = model_codes(model_dir, paths[seed], config.device)
        assert len(codes) == n_test
        assert (
            test.index["nucleus_id"].to_numpy()
            == pd.read_parquet(
                ref_dir / "activations" / "test" / "index.parquet",
                columns=["nucleus_id"],
            )["nucleus_id"].to_numpy()
        ).all()
        shares = contribution_shares(codes, w)
        stability["logit_weight"] = w
        stability["sign"] = np.sign(w).astype(int)
        stability["share_abs_phi"] = shares
        stability["activation_frequency"] = (codes > 0).mean(0)
        stability.insert(0, "model", name)
        stability.insert(1, "seed", seed)
        stability_rows.append(stability)
        np.save(out / f"codes_test_{name}.npy", codes)

        completeness = pd.read_csv(model_dir / "completeness.csv")
        completeness = completeness[
            (completeness["method"] == "sae") & (completeness["k"] == k)
        ]
        for row in completeness.itertuples(index=False):
            faithfulness.append(
                {
                    "model": name,
                    "run_id": entry.run_id,
                    "seed": int(row.seed),
                    "l1_coeff": row.l1_coeff,
                    "chosen": int(row.seed) == seed,
                    **{c: getattr(row, c) for c in FAITHFULNESS},
                }
            )
        signatures[name] = signature_table(codes, features, config.top_n)
        models[name] = (np.asarray(codes[sub], dtype=np.float64), stability)
        codes_w[name] = w
        del codes

    # reference row of tab:res-faithfulness, for the table
    ref_completeness = pd.read_csv(ref_dir / "completeness.csv")
    ref_completeness = ref_completeness[
        (ref_completeness["method"] == "sae") & (ref_completeness["k"] == k)
    ]
    for row in ref_completeness.itertuples(index=False):
        faithfulness.append(
            {
                "model": "reference",
                "run_id": config.parent_run_id,
                "seed": int(row.seed),
                "l1_coeff": row.l1_coeff,
                "chosen": int(row.seed) == ref_meta["seed"],
                **{c: getattr(row, c) for c in FAITHFULNESS},
            }
        )
    faithfulness = pd.DataFrame(faithfulness)
    faithfulness.to_csv(out / "faithfulness.csv", index=False)

    all_signatures = pd.concat(
        [s.assign(model=m, concept=np.arange(len(s))) for m, s in signatures.items()]
    )
    all_signatures.to_csv(out / "signatures.csv", index=False)
    scale = np.sqrt((all_signatures[list(SIGNATURE)] ** 2).mean()).to_numpy()

    matches, correlations = [], []
    for name, (sub_codes, stability) in models.items():
        table, corr = match_concepts(
            ref_sub,
            sub_codes,
            letters,
            ref_w,
            codes_w[name],
            signatures["reference"],
            signatures[name],
            scale,
            stability["stable"].to_numpy(),
            stability["share_abs_phi"].to_numpy(),
            config.match.min_correlation,
        )
        table.insert(0, "model", name)
        matches.append(table)
        correlations.append(
            pd.DataFrame(corr, columns=[f"c{j}" for j in range(corr.shape[1])]).assign(
                model=name, reference_concept=np.arange(len(corr))
            )
        )
        print(f"\n{name}\n{table.round(3).to_string(index=False)}")
        logger.log_metrics(
            {
                f"{name}/matched": int(table["matched"].sum()),
                f"{name}/stable": int(stability["stable"].sum()),
            }
        )
    matches = pd.concat(matches, ignore_index=True)
    matches.to_csv(out / "matching.csv", index=False)
    pd.concat(correlations, ignore_index=True).to_csv(
        out / "correlations.csv", index=False
    )
    stability = pd.concat(stability_rows, ignore_index=True)
    stability.to_csv(out / "concepts.csv", index=False)

    both = matches.groupby("letter")["matched"].all()
    summary = pd.DataFrame({"letter": both.index, "matched_in_all": both.to_numpy()})
    summary.to_csv(out / "summary.csv", index=False)
    print(
        f"\nA-H matched in all new models: {int(both.sum())} of {len(both)}: {', '.join(both.index[both])}"
    )
    logger.log_metrics({"matched_in_all": int(both.sum())})
    for path in (
        "faithfulness.csv",
        "signatures.csv",
        "matching.csv",
        "correlations.csv",
        "concepts.csv",
        "summary.csv",
    ):
        logger.log_artifact(str(out / path), "cross_model")


if __name__ == "__main__":
    main()
