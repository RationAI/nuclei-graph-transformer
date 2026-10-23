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
  `flip_margin`;
- slide level, two fixed aggregators of the nucleus logits of each test slide: the mean
  (`slide_mean_*`) and the mean of the top `slide_top_share` (1 %, at least one nucleus;
  `slide_top1pct_*`), F from f and F_A from f_A (its own top nuclei). AUROC / AUPRC against
  the slide label (`is_carcinoma`) of F and F_A, their differences F_A - F (`dAUROC`,
  `dAUPRC`) and Spearman of F with F_A. No slide-level sign flips (no fixed threshold).
- not reported: the model's attention pooling Σ_i softmax(s_i) f_i, with the attention
  head of `checkpoint` and the scores of h or of ĥ, in `slide_attention_untrained.csv`.
  The nucleus-level model was trained on nucleus labels only, so its attention head never
  received a gradient (still at initialization; near-uniform weights).

The OLS head is fitted from sparse sufficient statistics (`RidgeStatistics` with
alpha = 0, as `metrics.fit_ols`); the ridge head of `xai.evaluate_sae` is left out.
Every row is checked for MCE ≤ FE ≤ L_g·RE and |ATE - FE| < `ate_atol`; the run fails
after all outputs are saved if a check fails. The k = 16 SAE is re-evaluated here on the
CPU (fp32) and compared with its row of `completeness.csv` (GPU, TF32).

Codes of the main configuration are stored sparsely (latent indices and values of
the K active latents per nucleus, rows = `activations/test/index.parquet`) in
`codes/<model>/codes_test_{indices,values}.npy`.

Seeds: `completeness_summary.csv` gives mean, min, max and range over seeds per
configuration; `comparisons.csv` the differences of `comparisons` with `within_spread`
= |difference| ≤ the larger seed range of the two (single-seed pairs: the main
configuration's range).
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib
import mlflow


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy.stats import spearmanr

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import Activations, load_activations
from nuclei_graph.xai.metrics import (
    CompletenessAccumulator,
    LinearHead,
    RidgeHead,
    RidgeStatistics,
    classification_metrics,
    logit_agreement,
    rmse,
)
from nuclei_graph.xai.runtime import setup_precision
from nuclei_graph.xai.sae import ConceptAutoencoder, encode, load_autoencoder
from nuclei_graph.xai.topk import TopKAutoencoder, dense_codes, encode_topk, load_topk
from xai.concept_masks import code_commit
from xai.evaluate_sae import head_logits, recalibrated_fidelity, reference_statistics


type SparseCodes = tuple[NDArray[np.int16], NDArray[np.float32]]

CHECKS = ["check_MCE_le_FE", "check_FE_le_L_g_RE", "check_ATE_eq_FE"]
FREQUENCY_QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
SUMMARY_COLS = [
    "RE",
    "explained_variance_test",
    "FE",
    "FE_over_std_f",
    "MCE",
    "L_g_RE",
    "R2_FE",
    "spearman_f_fA",
    "flip_rate",
    "flip_rate_small_logit",
    "AUROC_fA",
    "AUPRC_fA",
    "nucleus_dAUROC",
    "nucleus_dAUPRC",
    "slide_mean_AUROC_FA",
    "slide_mean_dAUROC",
    "slide_mean_dAUPRC",
    "slide_mean_spearman_F_FA",
    "slide_top1pct_AUROC_FA",
    "slide_top1pct_dAUROC",
    "slide_top1pct_dAUPRC",
    "slide_top1pct_spearman_F_FA",
    "L0",
    "dead_fraction",
]
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


@dataclass(frozen=True)
class AttentionHead:
    """The model's `attn_head`: s(h) = W_2 tanh(W_1 h + b_1) + b_2."""

    w1: NDArray[np.float32]
    b1: NDArray[np.float32]
    w2: NDArray[np.float32]
    b2: float

    def __call__(self, h: NDArray, chunk_size: int = 262144) -> NDArray[np.float64]:
        return np.concatenate(
            [
                np.tanh(
                    np.asarray(h[i : i + chunk_size], np.float32) @ self.w1.T + self.b1
                )
                @ self.w2
                + self.b2
                for i in range(0, len(h), chunk_size)
            ]
        ).astype(np.float64)

    @classmethod
    def from_checkpoint(cls, uri: str, head: LinearHead) -> "AttentionHead":
        """Downloads the checkpoint; asserts that its class head is `head`."""
        with tempfile.TemporaryDirectory() as tmp:
            path = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp)
            state = torch.load(path, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
        np.testing.assert_allclose(
            state["net.class_head.weight"].double().numpy().ravel(), head.w, atol=1e-6
        )
        weights = {
            k: state[f"net.attn_head.{k}"].float().numpy()
            for k in ("0.weight", "0.bias", "2.weight", "2.bias")
        }
        return cls(
            weights["0.weight"],
            weights["0.bias"],
            weights["2.weight"].ravel(),
            float(weights["2.bias"][0]),
        )


def slide_pool(
    values: NDArray, scores: NDArray, starts: NDArray[np.integer]
) -> NDArray[np.float64]:
    """Σ softmax(scores) · values per slide; slides are contiguous from `starts`."""
    lengths = np.diff(np.append(starts, len(values)))
    scores = scores - np.repeat(np.maximum.reduceat(scores, starts), lengths)
    weights = np.exp(scores)
    return np.add.reduceat(weights * values, starts) / np.add.reduceat(weights, starts)


def slide_top_mean(
    values: NDArray, starts: NDArray[np.integer], share: float
) -> NDArray[np.float64]:
    """Mean of the top `share` (at least one) of the values of each slide."""
    ends = np.append(starts[1:], len(values))
    out = np.empty(len(starts))
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        n = end - start
        top = max(1, int(np.ceil(share * n)))
        out[i] = np.partition(values[start:end], n - top)[n - top :].mean()
    return out


def slide_metrics(
    f_slide: NDArray, f_a_slide: NDArray, labels: NDArray, prefix: str
) -> dict[str, float]:
    """AUROC / AUPRC of F and F_A, F_A - F, and Spearman of F with F_A."""
    original = classification_metrics(labels, f_slide)
    reconstructed = classification_metrics(labels, f_a_slide)
    row = {}
    for key in original:
        row[f"{prefix}_{key}_F"] = original[key]
        row[f"{prefix}_{key}_FA"] = reconstructed[key]
        row[f"{prefix}_d{key}"] = reconstructed[key] - original[key]
    row[f"{prefix}_spearman_F_FA"] = float(spearmanr(f_slide, f_a_slide).statistic)
    row[f"{prefix}_FE"] = rmse(f_slide, f_a_slide)
    return row


@dataclass(frozen=True)
class Slides:
    """Test slides in index order: first rows, labels, the attention scores of h."""

    starts: NDArray[np.int64]
    labels: NDArray[np.int64]
    attention_scores: NDArray[np.float64]  # untrained head, not reported
    top_share: float

    def aggregate(self, f: NDArray) -> dict[str, NDArray[np.float64]]:
        lengths = np.diff(np.append(self.starts, len(f)))
        return {
            "slide_mean": np.add.reduceat(f, self.starts) / lengths,
            "slide_top1pct": slide_top_mean(f, self.starts, self.top_share),
        }


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
    attention: AttentionHead,
    slides: Slides,
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

    U32, b_d32 = U.astype(np.float32), b_d.astype(np.float32)
    scores_hat = np.concatenate(
        [
            attention(
                dense_codes(*(c[s] for c in codes["test"]), num_latents) @ U32 + b_d32
            )
            for s in data["test"].chunks(chunk_size)
        ]
    )
    original, reconstructed = slides.aggregate(f["test"]), slides.aggregate(f_a)
    for prefix in original:
        row |= slide_metrics(
            original[prefix], reconstructed[prefix], slides.labels, prefix
        )
    for prefix, scores in (
        ("untrained_attn_h", slides.attention_scores),
        ("untrained_attn_hhat", scores_hat),
    ):
        row |= slide_metrics(
            slide_pool(f["test"], slides.attention_scores, slides.starts),
            slide_pool(f_a, scores, slides.starts),
            slides.labels,
            prefix,
        )
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


def group_name(row: pd.Series) -> str:
    if row["method"] == "sae_l1":
        return f"sae_l1_m{row['num_latents']}"
    return f"topk_m{row['num_latents']}_k{int(row['k'])}"


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, min, max and range over seeds per configuration."""
    grouped = df.groupby("group", sort=False)[SUMMARY_COLS]
    summary = pd.concat(
        {stat: grouped.agg(stat) for stat in ("mean", "min", "max")}, axis=1
    )
    summary.columns = [f"{col}_{stat}" for stat, col in summary.columns]
    for col in SUMMARY_COLS:
        summary[f"{col}_range"] = summary[f"{col}_max"] - summary[f"{col}_min"]
    summary["seeds"] = df.groupby("group", sort=False).size()
    return summary.reset_index()


def comparisons(
    summary: pd.DataFrame, pairs: list[list[str]], main_group: str
) -> pd.DataFrame:
    """Differences of configuration means and whether they are within the seed spread.

    Within: |difference| ≤ the larger seed range of the two configurations, or the main
    configuration's range if both have a single seed.
    """
    summary = summary.set_index("group")
    rows = []
    for (a, b), col in ((pair, col) for pair in pairs for col in SUMMARY_COLS):
        if a not in summary.index or b not in summary.index:
            continue
        spread = max(summary.loc[a, f"{col}_range"], summary.loc[b, f"{col}_range"])
        if max(summary.loc[a, "seeds"], summary.loc[b, "seeds"]) == 1:
            spread = summary.loc[main_group, f"{col}_range"]
        diff = summary.loc[a, f"{col}_mean"] - summary.loc[b, f"{col}_mean"]
        rows.append(
            {
                "a": a,
                "b": b,
                "metric": col,
                "mean_a": summary.loc[a, f"{col}_mean"],
                "mean_b": summary.loc[b, f"{col}_mean"],
                "difference": diff,
                "seed_spread": spread,
                "within_spread": abs(diff) <= spread,
            }
        )
    return pd.DataFrame(rows)


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

    attention = AttentionHead.from_checkpoint(config.checkpoint, head)
    slide_ids = data["test"].index["slide_id"].astype(str).to_numpy()
    starts = np.flatnonzero(np.r_[True, slide_ids[1:] != slide_ids[:-1]])
    assert len(np.unique(slide_ids)) == len(starts), "test slides are not contiguous"
    is_carcinoma = pd.read_parquet(config.slides).set_index("slide_id")["is_carcinoma"]
    scores = attention(data["test"].h)
    slides = Slides(
        starts=starts,
        labels=is_carcinoma.loc[slide_ids[starts]].to_numpy(np.int64),
        attention_scores=scores,
        top_share=config.slide_top_share,
    )
    slide_model_metrics = {
        f"{prefix}_{key}_F": value
        for prefix, values in slides.aggregate(f["test"]).items()
        for key, value in classification_metrics(slides.labels, values).items()
    }
    attention_pooled = slide_pool(f["test"], scores, starts)
    lengths = np.diff(np.append(starts, len(scores)))
    weights = np.exp(scores - np.repeat(np.maximum.reduceat(scores, starts), lengths))
    weights /= np.repeat(np.add.reduceat(weights, starts), lengths)
    effective = 1 / np.add.reduceat(weights**2, starts) / lengths
    slide_stats = {
        "slides": len(starts),
        "carcinoma_slides": int(slides.labels.sum()),
        **slide_model_metrics,
    }
    attention_stats = {  # untrained head, not reported
        "attention_effective_nuclei_share_median": float(np.median(effective)),
        "attention_effective_nuclei_share_min": float(effective.min()),
        "pooled_minus_mean_logit_median": float(
            np.median(attention_pooled - np.add.reduceat(f["test"], starts) / lengths)
        ),
    }
    print(slide_stats, attention_stats)
    logger.log_metrics({f"test/slides/{k}": v for k, v in slide_stats.items()})
    logger.log_metrics(
        {f"test/untrained_attention/{k}": v for k, v in attention_stats.items()}
    )
    reference = reference_statistics(data, f, holdout)
    logger.log_metrics({f"reference/{key}": value for key, value in reference.items()})

    grid = pd.read_csv(topk_dir / "training_grid.csv")
    entries = [
        (f"k16_l1_seed{seed}", path) for seed, path in config.reference_models.items()
    ]
    entries += [
        (row.model, str(topk_dir / row.path))
        for row in grid[grid["method"] == "topk"].itertuples(index=False)
    ]
    rows, frequencies = [], {}
    for name, path in entries:
        if name.startswith("k16_l1_"):
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
            model,
            head,
            data,
            f,
            holdout,
            config.chunk_size,
            config.flip_margin,
            attention,
            slides,
        )
        rows.append(
            {
                "model": name,
                **info,
                **row,
                **{f"{key}_f": value for key, value in model_metrics.items()},
                **slide_model_metrics,
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
    df["nucleus_dAUROC"] = df["AUROC_fA"] - df["AUROC_f"]
    df["nucleus_dAUPRC"] = df["AUPRC_fA"] - df["AUPRC_f"]
    untrained = [c for c in df.columns if c.startswith("untrained_attn")]
    df[["model", "method", "num_latents", "k", "seed", *untrained]].assign(
        note="untrained head, not reported", **attention_stats
    ).to_csv(topk_dir / "slide_attention_untrained.csv", index=False)
    df = df.drop(columns=untrained)
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
    df.insert(1, "group", df.apply(group_name, axis=1))
    df.to_csv(config.output_path, index=False)
    summary = seed_summary(df)
    summary.to_csv(topk_dir / "completeness_summary.csv", index=False)
    main_group = f"topk_m{config.main.num_latents}_k{config.main.k}"
    pairs = comparisons(summary, OmegaConf.to_container(config.comparisons), main_group)
    pairs.to_csv(topk_dir / "comparisons.csv", index=False)
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
    stored = stored[(stored["method"] == "sae") & (stored["k"] == 16)].set_index("seed")
    recomputed = df[df["method"] == "sae_l1"].set_index("seed")
    check = {
        f"seed{seed}": {
            col: {
                "stored": float(stored.loc[seed, col]),
                "recomputed": float(recomputed.loc[seed, col]),
            }
            for col in REFERENCE_COLS
        }
        for seed in recomputed.index
    }
    check["mean_over_seeds"] = {
        col: {
            "stored": float(stored.loc[recomputed.index, col].mean()),
            "recomputed": float(recomputed[col].mean()),
        }
        for col in REFERENCE_COLS
    }
    (topk_dir / "reference_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df[[*first, *CHECKS]])
    for row in df.to_dict("records"):
        logger.log_metrics(
            {
                f"test/{row['model']}/{key}": float(value)
                for key, value in row.items()
                if key not in ("model", "group", "method") and np.isfinite(float(value))
            }
        )
    for name in (
        config.output_path,
        topk_dir / "completeness_summary.csv",
        topk_dir / "comparisons.csv",
        topk_dir / "slide_attention_untrained.csv",
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
