"""Completeness of the overcomplete TopK SAEs on all test nuclei, next to the k = 16 SAE.

The same metrics and code as `xai.evaluate_sae` (`CompletenessAccumulator`): RE, FE,
MCE (OLS head from codes to logits, fitted on all validation nuclei), L_g·RE = ‖w‖·RE,
ATE, explained variance of h (holdout and test), FE / std(f), R²_FE, R²_MCE, logit
agreement (Pearson, Spearman, std(f_A) / std(f)), recalibrated fidelity, AUROC and
AUPRC of f_A next to f, L0 and dead fraction. All in the units of h: the TopK models
undo their input normalization in `decoder_weight` / `decoder_bias`. Added here:

- latent frequencies on test (share of nuclei where a latent is active, z_l > 0):
  quantiles per model, every latent in `latent_frequency_test.csv`;
- class flips: share of nuclei with sign(f) != sign(f_A), overall and among |f| <
  `flip_margin`.

The OLS head is fitted from sparse sufficient statistics (`RidgeStatistics` with
alpha = 0, as `metrics.fit_ols`); the ridge head of `xai.evaluate_sae` is left out.
Every row is checked for MCE ≤ FE ≤ L_g·RE and |ATE - FE| < `ate_atol`; the run fails
after all outputs are saved if a check fails. The k = 16 SAE is re-evaluated here on the
CPU (fp32) and compared with its row of `completeness.csv` (GPU, TF32).

Codes of the main configuration are stored sparsely (latent indices and values of
the K active latents per nucleus, rows = `activations/test/index.parquet`) in
`codes/<model>/codes_test_{indices,values}.npy`.
"""

import json
from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import Activations, load_activations
from nuclei_graph.xai.metrics import (
    CompletenessAccumulator,
    LinearHead,
    RidgeHead,
    RidgeStatistics,
    classification_metrics,
    logit_agreement,
)
from nuclei_graph.xai.runtime import setup_precision
from nuclei_graph.xai.sae import ConceptAutoencoder, encode, load_autoencoder
from nuclei_graph.xai.topk import TopKAutoencoder, dense_codes, encode_topk, load_topk
from xai.concept_masks import code_commit
from xai.evaluate_sae import head_logits, recalibrated_fidelity, reference_statistics


type SparseCodes = tuple[NDArray[np.int16], NDArray[np.float32]]

CHECKS = ["check_MCE_le_FE", "check_FE_le_L_g_RE", "check_ATE_eq_FE"]
FREQUENCY_QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
REFERENCE_COLS = [
    "RE",
    "FE",
    "MCE",
    "ATE",
    "L0",
    "AUROC_fA",
    "AUPRC_fA",
    "std_ratio_fA_f",
]


def sparse_codes(model: ConceptAutoencoder, h: NDArray) -> SparseCodes:
    """(indices, values) of the active-latent slots; a dense model keeps all k slots."""
    if isinstance(model, TopKAutoencoder):
        return encode_topk(model, h)
    codes = encode(model, h).numpy()
    indices = np.broadcast_to(np.arange(codes.shape[1], dtype=np.int16), codes.shape)
    return np.ascontiguousarray(indices), codes


def ols_head(codes: SparseCodes, num_latents: int, f: NDArray) -> RidgeHead:
    """`metrics.fit_ols` from sparse codes: the same sufficient statistics and solve."""
    indices, values = codes
    n, k = indices.shape
    x = scipy.sparse.csr_matrix(
        (
            values.ravel().astype(np.float64),
            indices.ravel(),
            np.arange(0, n * k + 1, k),
        ),
        shape=(n, num_latents),
    )
    stats = RidgeStatistics(num_latents)
    stats.n = n
    stats.sx = np.asarray(x.sum(0)).ravel()
    stats.sy = float(f.sum())
    stats.sxx = (x.T @ x).toarray()
    stats.sxy = x.T @ np.asarray(f, np.float64)
    return RidgeHead.from_statistics(stats, 0.0)


def accumulate(
    accumulator: CompletenessAccumulator,
    activations: Activations,
    f: NDArray,
    codes: SparseCodes,
    rows: NDArray[np.integer],
    chunk_size: int,
) -> dict[str, float]:
    num_latents = len(accumulator.decoder_weight)
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        dense = dense_codes(codes[0][chunk], codes[1][chunk], num_latents)
        accumulator.update(f[chunk], activations.z(chunk), dense)
    return accumulator.compute()


def latent_frequency(codes: SparseCodes, num_latents: int) -> NDArray[np.float64]:
    indices, values = codes
    counts = np.bincount(indices[values > 0].astype(np.int64), minlength=num_latents)
    return counts / len(indices)


def flip_rates(f: NDArray, f_a: NDArray, margin: float) -> dict[str, float]:
    flip = (f > 0) != (f_a > 0)
    small = np.abs(f) < margin
    return {
        "flip_rate": float(flip.mean()),
        "flip_rate_small_logit": float(flip[small].mean()),
        "share_small_logit": float(small.mean()),
    }


def evaluate_model(
    model: ConceptAutoencoder,
    head: LinearHead,
    data: dict[str, Activations],
    f: dict[str, NDArray],
    holdout: NDArray[np.bool_],
    chunk_size: int,
    flip_margin: float,
) -> tuple[dict[str, float], dict[str, SparseCodes], NDArray[np.float64]]:
    U = model.decoder_weight.detach().double().numpy()
    b_d = model.decoder_bias.detach().double().numpy()
    num_latents = len(U)
    codes = {split: sparse_codes(model, a.h) for split, a in data.items()}
    eta = ols_head(codes["validation"], num_latents, f["validation"])

    def run(split: str, rows: NDArray[np.integer]) -> CompletenessAccumulator:
        accumulator = CompletenessAccumulator(U, b_d, head, eta)
        accumulate(accumulator, data[split], f[split], codes[split], rows, chunk_size)
        return accumulator

    validation = run("validation", np.arange(len(data["validation"])))
    holdout_re = run("validation", np.flatnonzero(holdout)).compute()["RE"]
    test = run("test", np.arange(len(data["test"])))
    f_a = test.reconstructed
    row = test.compute()
    row |= logit_agreement(f["test"], f_a)
    row |= flip_rates(f["test"], f_a, flip_margin)
    row |= recalibrated_fidelity(
        validation.reconstructed, f["validation"], f_a, f["test"], row["FE"]
    )
    row["RE_holdout"] = holdout_re
    row |= {
        f"{key}_fA": value
        for key, value in classification_metrics(data["test"].labels, f_a).items()
    }
    validation_metrics = validation.compute()
    row["FE_validation"] = validation_metrics["FE"]
    row["MCE_validation"] = validation_metrics["MCE"]
    frequency = latent_frequency(codes["test"], num_latents)
    row |= {
        f"frequency_q{int(q * 100):02d}": float(np.quantile(frequency, q))
        for q in FREQUENCY_QUANTILES
    }
    row["share_latents_frequency_below_1e-4"] = float((frequency < 1e-4).mean())
    row["share_latents_frequency_above_10pct"] = float((frequency > 0.1).mean())
    return row, codes, frequency


def add_relative(df: pd.DataFrame, reference: dict[str, float]) -> pd.DataFrame:
    """As `xai.evaluate_sae.add_relative`, without the ridge head; plus Pearson²."""
    df["explained_variance_holdout"] = (
        1 - df["RE_holdout"] ** 2 / reference["total_variance_holdout"]
    )
    df["explained_variance_test"] = 1 - df["RE"] ** 2 / reference["total_variance_test"]
    df["FE_over_std_f"] = df["FE"] / reference["std_f_test"]
    var_f = reference["std_f_test"] ** 2
    df["R2_FE"] = 1 - df["FE"] ** 2 / var_f
    df["R2_MCE"] = 1 - df["MCE"] ** 2 / var_f
    df["pearson2_f_fA"] = df["pearson_f_fA"] ** 2
    for key in ("mean_norm_h_test", "std_f_test"):
        df[key] = reference[key]
    return df


def add_checks(df: pd.DataFrame, ate_atol: float) -> pd.DataFrame:
    df["check_MCE_le_FE"] = df["MCE"] <= df["FE"]
    df["check_FE_le_L_g_RE"] = df["FE"] <= df["L_g_RE"]
    df["check_ATE_eq_FE"] = (df["ATE"] - df["FE"]).abs() < ate_atol
    return df


def plot_frequencies(frequencies: dict[str, NDArray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(-7, 0, 57)
    for name, frequency in frequencies.items():
        values = np.log10(np.clip(frequency, 1e-7, None))
        ax.hist(values, bins=bins, histtype="step", density=True, label=name)
    ax.set(
        xlabel="log10 latent frequency on test (share of nuclei active; dead at -7)",
        ylabel="density over latents",
    )
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@with_cli_args(["+xai=evaluate_topk_sae"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    for key, value in setup_precision(config.matmul_precision).items():
        client.set_tag(logger.run_id, key, value)
    commit, dirty = code_commit("xai", "nuclei_graph", "configs")
    client.set_tag(
        logger.run_id, "xai.code_commit", f"{commit}{' (dirty)' if dirty else ''}"
    )
    for key, value in config.links.items():
        client.set_tag(logger.run_id, f"xai.{key}", value)
    for source in (__file__, "configs/xai/evaluate_topk_sae.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")
    torch.set_num_threads(config.num_threads)
    topk_dir = Path(config.topk_dir)
    (topk_dir / "figures").mkdir(exist_ok=True)

    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    data = {
        split: load_activations(config.activations_dir, split)
        for split in ("validation", "test")
    }
    if config.limit_rows:  # debugging only
        n = config.limit_rows
        data = {
            s: Activations(a.h[:n], a.logits[:n], a.index.iloc[:n])
            for s, a in data.items()
        }
    f = {split: head_logits(head, activations) for split, activations in data.items()}
    holdout_ids = set(
        pd.read_csv(Path(config.sae_dir) / "holdout_slides.csv")["slide_id"]
    )
    holdout = (
        data["validation"].index["slide_id"].astype(str).isin(holdout_ids).to_numpy()
    )
    model_metrics = classification_metrics(data["test"].labels, f["test"])
    reference = reference_statistics(data, f, holdout)
    logger.log_metrics({f"reference/{key}": value for key, value in reference.items()})

    grid = pd.read_csv(topk_dir / "training_grid.csv")
    entries = [("k16_l1_reference", config.reference_model)]
    entries += [
        (row.model, str(topk_dir / row.path))
        for row in grid[grid["method"] == "topk"].itertuples(index=False)
    ]
    rows, frequencies = [], {}
    for name, path in entries:
        if name == "k16_l1_reference":
            model, meta = load_autoencoder(path)
            info = {
                "method": "sae_l1",
                "num_latents": 16,
                "k": np.nan,
                "seed": meta["seed"],
            }
        else:
            model, meta = load_topk(path)
            info = {
                "method": "topk",
                "num_latents": meta["num_latents"],
                "k": meta["k"],
                "seed": meta["seed"],
            }
        row, codes, frequency = evaluate_model(
            model, head, data, f, holdout, config.chunk_size, config.flip_margin
        )
        rows.append(
            {
                "model": name,
                **info,
                **row,
                **{f"{key}_f": value for key, value in model_metrics.items()},
            }
        )
        frequencies[name] = frequency
        main = (info["num_latents"], info["k"]) == (
            config.main.num_latents,
            config.main.k,
        )
        if main and not config.limit_rows:
            out = topk_dir / "codes" / name
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "codes_test_indices.npy", codes["test"][0])
            np.save(out / "codes_test_values.npy", codes["test"][1])
        print(
            {
                key: rows[-1][key]
                for key in ("model", "RE", "FE", "MCE", "L0", "dead_fraction")
            }
        )

    df = add_checks(add_relative(pd.DataFrame(rows), reference), config.ate_atol)
    first = [
        "model",
        "method",
        "num_latents",
        "k",
        "seed",
        "RE",
        "explained_variance_test",
        "FE",
        "MCE",
        "L_g_RE",
        "ATE",
        "AUROC_f",
        "AUROC_fA",
        "AUPRC_f",
        "AUPRC_fA",
        "std_ratio_fA_f",
        "R2_FE",
        "pearson2_f_fA",
        "spearman_f_fA",
        "L0",
        "dead_fraction",
        "flip_rate",
        "flip_rate_small_logit",
    ]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(config.output_path, index=False)
    pd.DataFrame(
        [
            {"model": name, "latent": j, "frequency": value}
            for name, freq in frequencies.items()
            for j, value in enumerate(freq)
        ]
    ).to_csv(topk_dir / "latent_frequency_test.csv", index=False)
    plot_frequencies(frequencies, topk_dir / "figures" / "latent_frequency_test.png")

    # the k = 16 SAE: this CPU evaluation against xai.evaluate_sae (GPU, TF32)
    stored = pd.read_csv(config.reference_completeness)
    stored = stored[
        (stored["method"] == "sae")
        & (stored["k"] == 16)
        & (stored["seed"] == df.loc[0, "seed"])
    ].iloc[0]
    check = {
        col: {"stored": float(stored[col]), "recomputed": float(df.loc[0, col])}
        for col in REFERENCE_COLS
    }
    (topk_dir / "reference_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df[[*first, *CHECKS]])
    for row in df.to_dict("records"):
        logger.log_metrics(
            {
                f"{row['model']}/{key}": float(value)
                for key, value in row.items()
                if key not in ("model", "method") and np.isfinite(float(value))
            }
        )
    for name in (
        config.output_path,
        topk_dir / "latent_frequency_test.csv",
        topk_dir / "reference_check.json",
        topk_dir / "figures" / "latent_frequency_test.png",
    ):
        logger.log_artifact(str(name), "completeness")

    violations = df.loc[~df[CHECKS].all(axis=1), ["model", *CHECKS]]
    logger.log_metrics({"check_violations": len(violations)})
    assert violations.empty, f"Completeness checks violated:\n{violations.to_string()}"


if __name__ == "__main__":
    main()
