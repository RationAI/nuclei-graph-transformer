"""Completeness evaluation of the selected concept autoencoders on all test nuclei.

For every selected SAE (one λ per k and seed, see `xai.train_sae`) and every PCA
baseline: RE, FE, MCE, L_g·RE, ATE, the relative L1 reconstruction error of h mean|z - ẑ| / mean|z|, L0
and dead fraction, and AUROC/AUPRC of f_A next to those of f, on all test nuclei,
unweighted (the natural distribution).

f is the head applied to the stored representations, f = g(z); its maximal deviation
from the model's own logits (float16 storage of z) is reported separately.

MCE uses a head η from codes to logits fitted on all validation codes and evaluated
on test: `MCE` with ordinary least squares, `MCE_ridge` with ridge regression (alpha from
`RIDGE_ALPHAS`, selected on the holdout validation slides). MCE estimates the
infimum over heads; unlike FE ≤ L_g·RE, MCE ≤ FE is not guaranteed on test. Every
row is checked for MCE ≤ FE (both heads) ≤ L_g·RE and |ATE - FE| < `ate_atol`;
violations are written to the table and reported, and the run fails after all
outputs are saved.

Definitions (test, unweighted; z = h the representation, f the logit):
- relative L1 reconstruction error of h = Σ_i Σ_d |z_id - ẑ_id| / Σ_i Σ_d |z_id| (all nuclei i, dims d);
- explained variance of h = 1 - RE² / E‖z - mean(z)‖², on the holdout and on test;
- logit variance explained R²_FE = 1 - FE² / var(f), R²_MCE = 1 - MCE² / var(f);
- Pearson and Spearman correlation of f with f_A;
- shrinkage: f ≈ intercept + slope · f_A (least squares), and std(f_A) / std(f);
- recalibrated fidelity: f ≈ a · f_A + b fitted by least squares on all validation
  nuclei and applied on test, `FE_recalibrated` = RMSE(f, a · f_A + b) and
  `FE2_share_recalibration` = (FE² - FE_recalibrated²) / FE², the share of FE²
  removed by the recalibration.

Also: `completeness_summary.csv` (mean ± std over seeds), `stability.csv`
(Hungarian matching of seed-0 decoder directions to seeds 1 and 2, by cosine) and
`pca_spectrum.csv` (cumulative explained variance of validation h).
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import Activations, load_activations
from nuclei_graph.xai.metrics import (
    CompletenessAccumulator,
    LinearHead,
    classification_metrics,
    fit_ols,
    fit_ridge,
    logit_agreement,
    rmse,
)
from nuclei_graph.xai.runtime import setup_precision
from nuclei_graph.xai.sae import (
    CHUNK_SIZE,
    ConceptAutoencoder,
    covariance,
    encode,
    explained_variance,
    load_autoencoder,
)
from nuclei_graph.xai.stability import matched_cosines


COMPLETENESS_COLS = ["RE", "FE", "MCE", "L_g_RE", "ATE", "relative_l1_error_h"]
RELATIVE_COLS = [
    "explained_variance_holdout",
    "explained_variance_test",
    "FE_over_std_f",
    "R2_FE",
    "R2_MCE",
    "R2_MCE_ridge",
]
RECALIBRATION_COLS = [
    "FE_recalibrated",
    "FE2_share_recalibration",
    "recalibration_a",
    "recalibration_b",
]
AGREEMENT_COLS = [
    "pearson_f_fA",
    "spearman_f_fA",
    "shrinkage_slope",
    "shrinkage_intercept",
    "std_ratio_fA_f",
]
CHECKS = [
    "check_MCE_le_FE",
    "check_MCE_ridge_le_FE",
    "check_FE_le_L_g_RE",
    "check_ATE_eq_FE",
]


def head_logits(head: LinearHead, activations: Activations) -> np.ndarray:
    return np.concatenate([head(activations.z(rows)) for rows in activations.chunks()])


def accumulate(
    accumulator: CompletenessAccumulator,
    activations: Activations,
    f: np.ndarray,
    codes: np.ndarray,
) -> dict[str, float]:
    for rows in activations.chunks():
        accumulator.update(f[rows], activations.z(rows), codes[rows])
    return accumulator.compute()


def reference_statistics(
    data: dict[str, Activations], f: dict[str, np.ndarray], holdout: np.ndarray
) -> dict[str, float]:
    """Total variances E‖h - mean(h)‖² (holdout, test), mean ‖h‖ and std of f on test."""
    holdout_h = np.asarray(data["validation"].h[np.flatnonzero(holdout)])
    test = data["test"]
    norms = np.concatenate(
        [np.linalg.norm(test.z(rows), axis=1) for rows in test.chunks()]
    )
    return {
        "total_variance_holdout": float(np.trace(covariance(holdout_h)[1])),
        "total_variance_test": float(np.trace(covariance(test.h)[1])),
        "mean_norm_h_test": float(norms.mean()),
        "std_f_test": float(f["test"].std()),
    }


def recalibrated_fidelity(
    f_a_validation: np.ndarray,
    f_validation: np.ndarray,
    f_a_test: np.ndarray,
    f_test: np.ndarray,
    fe: float,
) -> dict[str, float]:
    """F ≈ a · f_A + b by least squares on validation, evaluated on test."""
    a, b = np.polyfit(f_a_validation, f_validation, deg=1)
    fe_recalibrated = rmse(f_test, a * f_a_test + b)
    return {
        "FE_recalibrated": fe_recalibrated,
        "FE2_share_recalibration": (fe**2 - fe_recalibrated**2) / fe**2,
        "recalibration_a": float(a),
        "recalibration_b": float(b),
    }


def add_relative(df: pd.DataFrame, reference: dict[str, float]) -> pd.DataFrame:
    """Explained variance 1 - RE² / E‖h - mean(h)‖² and FE / std(f)."""
    df["explained_variance_holdout"] = (
        1 - df["RE_holdout"] ** 2 / reference["total_variance_holdout"]
    )
    df["explained_variance_test"] = 1 - df["RE"] ** 2 / reference["total_variance_test"]
    df["FE_over_std_f"] = df["FE"] / reference["std_f_test"]
    var_f = reference["std_f_test"] ** 2
    df["R2_FE"] = 1 - df["FE"] ** 2 / var_f
    df["R2_MCE"] = 1 - df["MCE"] ** 2 / var_f
    df["R2_MCE_ridge"] = 1 - df["MCE_ridge"] ** 2 / var_f
    for key in ("mean_norm_h_test", "std_f_test"):
        df[key] = reference[key]
    return df


def evaluate_model(
    model: ConceptAutoencoder,
    head: LinearHead,
    data: dict[str, Activations],
    f: dict[str, np.ndarray],
    holdout: np.ndarray,
    device: str,
) -> dict[str, float]:
    model = model.to(device)
    U = model.decoder_weight.detach().double().cpu().numpy()
    b_d = model.decoder_bias.detach().double().cpu().numpy()
    codes = {
        split: encode(model, activations.h).numpy()
        for split, activations in data.items()
    }

    codes_val = codes["validation"].astype(np.float64)
    eta = fit_ols(codes_val, f["validation"])
    extra = {"ridge": fit_ridge(codes_val, f["validation"], holdout)}
    del codes_val
    validation_accumulator = CompletenessAccumulator(U, b_d, head, eta, extra)
    validation = accumulate(
        validation_accumulator, data["validation"], f["validation"], codes["validation"]
    )
    holdout_accumulator = CompletenessAccumulator(U, b_d, head, eta)
    holdout_rows = np.flatnonzero(holdout)
    for start in range(0, len(holdout_rows), CHUNK_SIZE):
        rows = holdout_rows[start : start + CHUNK_SIZE]
        holdout_accumulator.update(
            f["validation"][rows],
            data["validation"].z(rows),
            codes["validation"][rows],
        )
    test_accumulator = CompletenessAccumulator(U, b_d, head, eta, extra)
    row = accumulate(test_accumulator, data["test"], f["test"], codes["test"])
    row |= logit_agreement(f["test"], test_accumulator.reconstructed)
    row |= recalibrated_fidelity(
        validation_accumulator.reconstructed,
        f["validation"],
        test_accumulator.reconstructed,
        f["test"],
        row["FE"],
    )
    row["RE_holdout"] = holdout_accumulator.compute()["RE"]

    row |= {
        f"{key}_fA": value
        for key, value in classification_metrics(
            data["test"].labels, test_accumulator.reconstructed
        ).items()
    }
    row["ridge_alpha"] = extra["ridge"].alpha
    row["FE_validation"] = validation["FE"]
    row["MCE_validation"] = validation["MCE"]
    return row


def add_checks(df: pd.DataFrame, ate_atol: float) -> pd.DataFrame:
    df["check_MCE_le_FE"] = df["MCE"] <= df["FE"]
    df["check_MCE_ridge_le_FE"] = df["MCE_ridge"] <= df["FE"]
    df["check_FE_le_L_g_RE"] = df["FE"] <= df["L_g_RE"]
    df["check_ATE_eq_FE"] = (df["ATE"] - df["FE"]).abs() < ate_atol
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds per (method, k)."""
    cols = [
        *COMPLETENESS_COLS,
        "MCE_ridge",
        *RELATIVE_COLS,
        *RECALIBRATION_COLS,
        *AGREEMENT_COLS,
        "L0",
        "dead_fraction",
        "AUROC_fA",
        "AUPRC_fA",
        "l1_coeff",
    ]
    summary = df.groupby(["method", "k"])[cols].agg(["mean", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary["seeds"] = df.groupby(["method", "k"]).size()
    return summary.reset_index()


def stability(models: pd.DataFrame, sae_dir: Path, threshold: float) -> pd.DataFrame:
    """Per k and concept of seed 0: cosine of its Hungarian match in the other seeds."""
    rows = []
    for k, group in models[models["method"] == "sae"].groupby("k"):
        paths = dict(zip(group["seed"], group["path"], strict=True))
        if 0 not in paths:
            print(f"[WARN] stability k={k}: seed 0 not selected, skipped")
            continue
        reference = load_autoencoder(str(sae_dir / paths[0]))[0].decoder_weight
        for seed, path in sorted(paths.items()):
            if seed == 0:
                continue
            other = load_autoencoder(str(sae_dir / path))[0].decoder_weight
            match, cosine = matched_cosines(
                reference.detach().numpy(), other.detach().numpy()
            )
            rows += [
                {
                    "k": k,
                    "seed": seed,
                    "concept_seed0": j,
                    "matched_concept": int(match[j]),
                    "cosine": float(cosine[j]),
                    "above_threshold": bool(cosine[j] > threshold),
                }
                for j in range(len(cosine))
            ]
    return pd.DataFrame(rows)


@with_cli_args(["+xai=evaluate_sae"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    for key, value in setup_precision(config.matmul_precision).items():
        logger.experiment.set_tag(logger.run_id, key, value)
    out_dir = Path(config.out_dir)
    sae_dir = Path(config.sae_dir)

    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    data = {
        split: load_activations(config.activations_dir, split)
        for split in ("validation", "test")
    }
    f = {split: head_logits(head, activations) for split, activations in data.items()}
    for split, activations in data.items():
        deviation = float(np.abs(f[split] - activations.logits).max())
        logger.log_metrics({f"{split}/head_vs_model_logits_max_abs": deviation})
        print(
            f"{split}: {len(activations):,} nuclei, max |g(z) - logits| = {deviation:.2e}"
        )

    holdout_ids = set(pd.read_csv(sae_dir / "holdout_slides.csv")["slide_id"])
    holdout = (
        data["validation"].index["slide_id"].astype(str).isin(holdout_ids).to_numpy()
    )

    model_metrics = classification_metrics(data["test"].labels, f["test"])
    logger.log_metrics({f"model/{key}": value for key, value in model_metrics.items()})

    # PCA spectrum of validation h: natural distribution
    spectrum = explained_variance(data["validation"].h)
    pca_spectrum = pd.DataFrame(
        {
            "n_components": config.spectrum_components,
            "cumulative_explained_variance": [
                spectrum[n - 1] for n in config.spectrum_components
            ],
        }
    )
    pca_spectrum.to_csv(out_dir / "pca_spectrum.csv", index=False)
    print(pca_spectrum.to_string(index=False))

    manifest = pd.read_csv(sae_dir / "models.csv")
    rows = []
    for entry in manifest.itertuples(index=False):
        model, meta = load_autoencoder(str(sae_dir / entry.path))
        rows.append(
            {
                "method": meta["method"],
                "k": meta["k"],
                "seed": meta["seed"],
                "l1_coeff": meta.get("l1_coeff", np.nan),
                **evaluate_model(model, head, data, f, holdout, config.device),
                **{f"{key}_f": value for key, value in model_metrics.items()},
            }
        )
        print(
            {key: rows[-1][key] for key in ("method", "k", "seed", "RE", "FE", "MCE")}
        )

    reference = reference_statistics(data, f, holdout)
    print(reference)
    logger.log_metrics({f"reference/{key}": value for key, value in reference.items()})
    df = add_checks(add_relative(pd.DataFrame(rows), reference), config.ate_atol)
    first = ["method", "k", "seed", "l1_coeff", *COMPLETENESS_COLS, "MCE_ridge"]
    first += [*RELATIVE_COLS, *RECALIBRATION_COLS, *AGREEMENT_COLS]
    first += ["L0", "dead_fraction", "AUROC_f", "AUROC_fA", "AUPRC_f", "AUPRC_fA"]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(config.output_path, index=False)

    # PCA baselines (weighted fit on the non-holdout validation slides) next to the
    # spectrum of an unweighted PCA on all validation h
    pca = df[df["method"] == "pca"].set_index("k")
    pca_spectrum["baseline_explained_variance_holdout"] = pca_spectrum[
        "n_components"
    ].map(pca["explained_variance_holdout"])
    pca_spectrum["baseline_explained_variance_test"] = pca_spectrum["n_components"].map(
        pca["explained_variance_test"]
    )
    pca_spectrum.to_csv(out_dir / "pca_spectrum.csv", index=False)
    print(pca_spectrum.to_string(index=False))
    summary = summarize(df)
    summary.to_csv(out_dir / "completeness_summary.csv", index=False)

    matches = stability(manifest, sae_dir, config.stability_threshold)
    matches.to_csv(out_dir / "stability.csv", index=False)
    if len(matches):
        shares = matches.groupby(["k", "seed"])["above_threshold"].mean()
        print(
            "share of seed-0 concepts with matched cosine >", config.stability_threshold
        )
        print(shares.to_string())
        for (k, seed), share in shares.items():
            logger.log_metrics({f"stability/k{k}/seed{seed}/share_above": float(share)})

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df[[*first, *CHECKS]])

    for row in df.to_dict("records"):
        prefix = f"{row['method']}_k{row['k']}/seed{row['seed']}"
        logger.log_metrics(
            {
                f"{prefix}/{key}": float(value)
                for key, value in row.items()
                if key not in ("method", "k", "seed") and np.isfinite(float(value))
            }
        )
    for name in (
        config.output_path,
        out_dir / "completeness_summary.csv",
        out_dir / "stability.csv",
        out_dir / "pca_spectrum.csv",
    ):
        logger.log_artifact(str(name), "completeness")

    violations = df.loc[~df[CHECKS].all(axis=1), ["method", "k", "seed", *CHECKS]]
    logger.log_metrics({"check_violations": len(violations)})
    assert violations.empty, f"Completeness checks violated:\n{violations.to_string()}"


if __name__ == "__main__":
    main()
