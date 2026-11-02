r"""Unpenalised magnitudes on the encoder's support: NNLS refit of the k = 16 seed-1 SAE codes.

For every nucleus i (validation and test) the set of active concepts S_i = {j : κ_ij > 0}
of the SAE encoder is kept and the magnitudes are refitted without the L1 penalty:

    κ̃_i = argmin_{κ ≥ 0, κ_j = 0 for j ∉ S_i} ‖h_i - b_D - Σ_j κ_j u_j‖²

(non-negative least squares on the columns u_j, j ∈ S_i). No retraining; the encoder, the
decoder and the stored codes are unchanged, κ̃ is written to `refit_dir`. κ and S_i are
the stored codes (GPU encoding, as used by all concept analyses; a CPU re-encoding is
compared with them in `checks.json`). The support can only shrink (an NNLS coefficient
may become 0).

Solver: per nucleus, the unconstrained least-squares solution on the passive set P (start
P = S_i), dropping coefficients ≤ 0 until none is negative; vectorised over nuclei with
the same P. The result is exact iff the KKT conditions hold (κ̃_P > 0, gradient ≥ 0 on
S_i \\ P); nuclei that fail are solved with `scipy.optimize.nnls` (Lawson-Hanson). Checked:
KKT on all nuclei, agreement with scipy on a random sample, and ‖r - Uᵀκ̃‖ ≤ ‖r - Uᵀκ‖
for every nucleus (the encoder's κ is feasible).

Metrics (as `xai.evaluate_topk_sae`, f the head's logit of h, f_A of the reconstruction;
on validation and test): RE, explained variance of h (1 - RE² / E‖h - mean(h)‖² of the
split), FE, FE / std(f), std(f_A) / std(f), Spearman of f with f_A, AUROC / AUPRC of f_A
(nucleus labels), class flips sign(f) ≠ sign(f_A) overall and among |f| < `flip_margin`,
MCE (OLS head on the validation codes of the same variant), L0. Next to them: the encoder
codes (L1) recomputed here, and the stored TopK m = 16 rows of `topk/completeness.csv`
(test only).

Importance: φ̃_ij = κ̃_ij wᵀu_j; mean |φ_j| = |wᵀu_j| · mean κ_j and its share of Σ_j, for
the encoder codes and for κ̃, per split; letters from the deck key.

Per concept (`overlap.csv`, the stable concepts first): Spearman of κ_j with κ̃_j over the
nuclei where the concept is active (κ_j > 0; κ̃_j may be 0), on test and validation; and
the overlap |T(κ) ∩ T(κ̃)| / `top_n` of the top-`top_n` test nuclei by κ_j and by κ̃_j,
without a per-slide limit and with at most `max_per_slide` per slide (the gallery rule of
`xai.concept_material.top_nuclei`, applied to all active nuclei).

Logged as a child run of the explained model: the tables, `checks.json` and the code; κ̃
(about 1.3 GB) is not logged, its path is in the tag `xai.refit_codes_dir`.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy.optimize import nnls
from scipy.stats import spearmanr

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import Activations, load_activations
from nuclei_graph.xai.metrics import (
    CompletenessAccumulator,
    LinearHead,
    classification_metrics,
    concept_logit_weights,
    fit_ols,
    logit_agreement,
)
from nuclei_graph.xai.sae import covariance, encode, load_autoencoder
from xai.concept_masks import code_commit
from xai.concept_morphometry import concept_letters
from xai.evaluate_sae import head_logits
from xai.evaluate_topk_sae import flip_rates


SPLITS = ("validation", "test")
REPORT_COLS = [
    "RE",
    "explained_variance",
    "FE",
    "FE_over_std_f",
    "std_ratio_fA_f",
    "spearman_f_fA",
    "AUROC_fA",
    "AUPRC_fA",
    "flip_rate",
    "flip_rate_small_logit",
    "MCE",
    "L0",
]


def solve_passive(
    gram: NDArray, b: NDArray, passive: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Unconstrained least squares on each row's passive set, grouped by pattern."""
    kappa = np.zeros_like(b)
    keys = passive @ (1 << np.arange(passive.shape[1]))
    for key in np.unique(keys):
        if key == 0:
            continue
        rows = np.flatnonzero(keys == key)
        cols = np.flatnonzero(passive[rows[0]])
        kappa[np.ix_(rows, cols)] = np.linalg.solve(
            gram[np.ix_(cols, cols)], b[np.ix_(rows, cols)].T
        ).T
    return kappa


def refit_chunk(
    r: NDArray, active: NDArray[np.bool_], decoder: NDArray, gram: NDArray, tol: float
) -> tuple[NDArray[np.float64], int]:
    """NNLS magnitudes on the support `active`; returns κ̃ and the number of fallbacks."""
    b = r @ decoder.T
    passive = active.copy()
    for _ in range(active.shape[1] + 1):
        kappa = solve_passive(gram, b, passive)
        negative = passive & (kappa <= 0)
        if not negative.any():
            break
        passive &= ~negative
    kappa = np.where(passive, np.maximum(kappa, 0.0), 0.0)
    grad = kappa @ gram - b  # of ½‖r - Uᵀκ‖²
    scale = tol * (1.0 + np.abs(b).max(1, keepdims=True))
    kkt = ((grad >= -scale) | ~active | passive).all(1)  # dual feasible off P
    kkt &= ((np.abs(grad) <= scale) | ~passive).all(1)  # stationary on P
    failed = np.flatnonzero(~kkt)
    for i in failed:
        cols = np.flatnonzero(active[i])
        kappa[i] = 0.0
        kappa[i, cols] = nnls(decoder[cols].T, r[i])[0]
    return kappa, len(failed)


def refit_split(
    activations: Activations,
    codes: NDArray,
    decoder: NDArray,
    b_d: NDArray,
    chunk_size: int,
    tol: float,
) -> tuple[NDArray[np.float32], dict[str, float]]:
    gram = decoder @ decoder.T
    out = np.zeros(codes.shape, np.float32)
    fallbacks, worse = 0, 0
    for start in range(0, len(codes), chunk_size):
        rows = slice(start, min(start + chunk_size, len(codes)))
        r = activations.z(rows) - b_d
        c = np.asarray(codes[rows], np.float64)
        kappa, n_failed = refit_chunk(r, c > 0, decoder, gram, tol)
        fallbacks += n_failed
        before = ((r - c @ decoder) ** 2).sum(1)
        after = ((r - kappa @ decoder) ** 2).sum(1)
        worse += int((after > before * (1 + 1e-9) + 1e-9).sum())
        out[rows] = kappa
    return out, {"fallbacks_scipy": fallbacks, "nuclei_with_larger_residual": worse}


def scipy_check(
    activations: Activations,
    codes: NDArray,
    refit: NDArray,
    decoder: NDArray,
    b_d: NDArray,
    n: int,
    seed: int,
) -> dict[str, float]:
    """κ̃ against scipy's Lawson-Hanson NNLS on a random sample of nuclei."""
    rows = np.sort(np.random.default_rng(seed).choice(len(codes), n, replace=False))
    r = activations.z(rows) - b_d
    diff, objective = 0.0, 0.0
    for k, i in enumerate(rows):
        cols = np.flatnonzero(codes[i] > 0)
        reference = np.zeros(decoder.shape[0])
        if len(cols):
            reference[cols] = nnls(decoder[cols].T, r[k])[0]
        diff = max(diff, float(np.abs(reference - refit[i]).max()))
        mine = ((r[k] - refit[i].astype(np.float64) @ decoder) ** 2).sum()
        theirs = ((r[k] - reference @ decoder) ** 2).sum()
        objective = max(objective, float(mine - theirs))
    return {
        "n": n,
        "max_abs_kappa_diff": diff,
        "max_objective_excess_over_scipy": objective,
    }


def evaluate(
    codes: dict[str, NDArray],
    data: dict[str, Activations],
    f: dict[str, NDArray],
    decoder: NDArray,
    b_d: NDArray,
    head: LinearHead,
    total_variance: dict[str, float],
    flip_margin: float,
    chunk_size: int,
) -> list[dict]:
    eta = fit_ols(np.asarray(codes["validation"], np.float64), f["validation"])
    rows = []
    for split in SPLITS:
        accumulator = CompletenessAccumulator(decoder, b_d, head, eta)
        for s in data[split].chunks(chunk_size):
            accumulator.update(f[split][s], data[split].z(s), codes[split][s])
        f_a = accumulator.reconstructed
        row = accumulator.compute()
        row |= logit_agreement(f[split], f_a)
        row |= flip_rates(f[split], f_a, flip_margin)
        row |= {
            f"{k}_fA": v
            for k, v in classification_metrics(data[split].labels, f_a).items()
        }
        row |= {
            f"{k}_f": v
            for k, v in classification_metrics(data[split].labels, f[split]).items()
        }
        row["explained_variance"] = 1 - row["RE"] ** 2 / total_variance[split]
        row["FE_over_std_f"] = row["FE"] / float(f[split].std())
        rows.append({"split": split, **row})
    return rows


def importance(
    codes: NDArray, logit_weights: NDArray, letters: dict[int, str], chunk_size: int
) -> pd.DataFrame:
    total = np.zeros(codes.shape[1])
    active = np.zeros(codes.shape[1])
    for start in range(0, len(codes), chunk_size):
        chunk = np.asarray(codes[start : start + chunk_size], np.float64)
        total += chunk.sum(0)
        active += (chunk > 0).sum(0)
    mean_abs_phi = np.abs(logit_weights) * total / len(codes)
    return pd.DataFrame(
        {
            "concept": np.arange(len(total)),
            "letter": [letters[j] for j in range(len(total))],
            "logit_weight": logit_weights,
            "frequency": active / len(codes),
            "mean_kappa": total / len(codes),
            "mean_abs_phi": mean_abs_phi,
            "share_abs_phi": mean_abs_phi / mean_abs_phi.sum(),
        }
    )


def top_rows(
    column: NDArray, slide: NDArray[np.int64], top_n: int, max_per_slide: int | None
) -> set[int]:
    """The `top_n` rows with the highest active κ_j, optionally capped per slide.

    The rule of `xai.concept_material.top_nuclei` (descending κ_j, at most
    `max_per_slide` per slide) over all active nuclei: its candidate pool of
    top_n · max_per_slide · 50 rows is too small for A and B at top_n = 2,000.
    """
    active = np.flatnonzero(column > 0)
    order = active[np.argsort(-column[active], kind="stable")]
    if max_per_slide is not None:
        within = pd.Series(slide[order]).groupby(slide[order]).cumcount().to_numpy()
        order = order[within < max_per_slide]
    return {int(i) for i in order[:top_n]}


def overlap_table(
    codes: dict[str, NDArray],
    refit: dict[str, NDArray],
    test: Activations,
    concepts: list[tuple[int, str]],
    stable: list[str],
    top_n: int,
    max_per_slide: int,
) -> pd.DataFrame:
    slide = test.slide_codes
    rows = []
    for j, letter in concepts:
        row: dict[str, object] = {
            "letter": letter,
            "concept": j,
            "stable": letter in stable,
        }
        for split in SPLITS:
            on = np.asarray(codes[split][:, j]) > 0
            kappa, kappa_refit = codes[split][on, j], refit[split][on, j]
            row[f"n_active_{split}"] = int(on.sum())
            row[f"spearman_kappa_refit_{split}"] = float(
                spearmanr(kappa, kappa_refit).statistic
            )
            row[f"share_refit_zero_{split}"] = float((kappa_refit == 0).mean())
        kappa, kappa_refit = (np.asarray(c["test"][:, j]) for c in (codes, refit))
        for suffix, cap in (("", None), ("_capped", max_per_slide)):
            a = top_rows(kappa, slide, top_n, cap)
            b = top_rows(kappa_refit, slide, top_n, cap)
            row[f"top{suffix}_n_l1"], row[f"top{suffix}_n_refit"] = len(a), len(b)
            row[f"top{suffix}_overlap_pct"] = 100 * len(a & b) / top_n
        rows.append(row)
    table = pd.DataFrame(rows)
    order = {letter: i for i, letter in enumerate(stable)}
    return table.sort_values(
        "letter", key=lambda s: s.map(lambda x: order.get(x, len(order))), kind="stable"
    ).reset_index(drop=True)


@with_cli_args(["+xai=refit_magnitudes"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    commit, dirty = code_commit("xai", "nuclei_graph", "configs")
    client.set_tag(
        logger.run_id, "xai.code_commit", f"{commit}{' (dirty)' if dirty else ''}"
    )
    for key, value in config.links.items():
        client.set_tag(logger.run_id, f"xai.{key}", value)
    client.set_tag(logger.run_id, "xai.refit_codes_dir", str(config.refit_dir))
    torch.set_num_threads(config.num_threads)
    out = Path(config.refit_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, meta = load_autoencoder(config.sae_model)
    assert (meta["k"], meta["seed"]) == (16, 1), meta
    decoder = model.decoder_weight.detach().double().numpy()
    b_d = model.decoder_bias.detach().double().numpy()
    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    data = {s: load_activations(config.activations_dir, s) for s in SPLITS}
    f = {s: head_logits(head, a) for s, a in data.items()}
    total_variance = {s: float(np.trace(covariance(a.h)[1])) for s, a in data.items()}

    checks: dict[str, object] = {}
    codes, refit = {}, {}
    for split in SPLITS:
        # the stored codes (GPU encoding, used by all concept analyses) are the support
        # and the L1 values; a CPU re-encoding is only compared with them
        codes[split] = np.load(config.stored_codes[split])
        assert codes[split].shape == (len(data[split]), decoder.shape[0])
        fresh = encode(model, data[split].h).numpy()
        checks[f"{split}/cpu_encoding_max_abs_diff_vs_stored"] = float(
            np.abs(fresh - codes[split]).max()
        )
        checks[f"{split}/cpu_encoding_support_mismatch_entries"] = int(
            ((fresh > 0) != (codes[split] > 0)).sum()
        )
        del fresh
        refit[split], info = refit_split(
            data[split], codes[split], decoder, b_d, config.chunk_size, config.kkt_tol
        )
        checks |= {f"{split}/{k}": v for k, v in info.items()}
        checks[f"{split}/scipy_sample"] = scipy_check(
            data[split],
            codes[split],
            refit[split],
            decoder,
            b_d,
            config.check_n,
            config.seed,
        )
        dropped = (codes[split] > 0) & (refit[split] == 0)
        checks[f"{split}/share_active_entries_set_to_zero"] = float(
            dropped.sum() / (codes[split] > 0).sum()
        )
        np.save(out / f"codes_refit_{split}.npy", refit[split])
        print(
            split, json.dumps({k: v for k, v in checks.items() if k.startswith(split)})
        )

    rows = []
    for variant, variant_codes in (("sae_l1", codes), ("sae_l1_nnls_refit", refit)):
        for row in evaluate(
            variant_codes,
            data,
            f,
            decoder,
            b_d,
            head,
            total_variance,
            config.flip_margin,
            config.chunk_size,
        ):
            rows.append({"variant": variant, **row})
    topk = pd.read_csv(config.topk_completeness)
    topk = topk[topk["group"] == config.topk_group].rename(
        columns={"explained_variance_test": "explained_variance"}
    )
    for _, t in topk.iterrows():
        rows.append(
            {
                "variant": f"{config.topk_group}_seed{t['seed']}",
                "split": "test",
                **{c: t[c] for c in [*REPORT_COLS, "AUROC_f", "AUPRC_f"]},
            }
        )
    rows.append(
        {
            "variant": f"{config.topk_group}_mean",
            "split": "test",
            **topk[[*REPORT_COLS, "AUROC_f", "AUPRC_f"]].mean().to_dict(),
        }
    )
    metrics = pd.DataFrame(rows)
    first = ["variant", "split", *REPORT_COLS, "AUROC_f", "AUPRC_f"]
    metrics = metrics[first + [c for c in metrics.columns if c not in first]]
    metrics.to_csv(out / "metrics.csv", index=False)

    # this recomputation against the stored L1 seed-1 row of topk/completeness.csv
    stored_l1 = (
        pd.read_csv(config.topk_completeness).set_index("model").loc["k16_l1_seed1"]
    )
    mine = metrics.query("variant == 'sae_l1' and split == 'test'").iloc[0]
    checks["l1_test_recomputed_vs_stored"] = {
        c: {"stored": float(stored_l1[c]), "recomputed": float(mine[c])}
        for c in (
            "RE",
            "FE",
            "std_ratio_fA_f",
            "spearman_f_fA",
            "AUROC_fA",
            "flip_rate",
        )
    }

    letters = dict(concept_letters(config.deck_key))
    weights = concept_logit_weights(decoder, head)
    parts = []
    for split in SPLITS:
        before = importance(codes[split], weights, letters, config.chunk_size)
        after = importance(refit[split], weights, letters, config.chunk_size)
        merged = before.merge(
            after[
                ["concept", "frequency", "mean_kappa", "mean_abs_phi", "share_abs_phi"]
            ],
            on="concept",
            suffixes=("_l1", "_refit"),
        )
        parts.append(merged.assign(split=split))
    shares = pd.concat(parts, ignore_index=True)
    shares["share_change_pp"] = 100 * (
        shares["share_abs_phi_refit"] - shares["share_abs_phi_l1"]
    )
    shares["kappa_ratio_refit_l1"] = (
        shares["mean_kappa_refit"] / shares["mean_kappa_l1"]
    )
    shares.sort_values(["split", "letter"]).to_csv(out / "importance.csv", index=False)
    key = pd.read_csv(config.deck_key).set_index("concept")["share_abs_phi"]
    test_l1 = shares.query("split == 'test'").set_index("concept")["share_abs_phi_l1"]
    checks["importance_l1_test_max_abs_diff_vs_deck_key"] = float(
        (test_l1 - key.loc[test_l1.index]).abs().max()
    )
    (out / "checks.json").write_text(json.dumps(checks, indent=2))

    concepts = concept_letters(config.deck_key)
    stable = list(config.stable_concepts)
    overlap = overlap_table(
        codes, refit, data["test"], concepts, stable, config.top_n, config.max_per_slide
    )
    overlap.to_csv(out / "overlap.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(metrics[first].to_string(index=False))
        main_letters = list(config.stable_concepts)
        view = shares[shares["letter"].isin(main_letters)].pivot(
            index="letter",
            columns="split",
            values=["share_abs_phi_l1", "share_abs_phi_refit", "kappa_ratio_refit_l1"],
        )
        print(view.round(4).to_string())
        print(overlap.to_string(index=False))
    print(json.dumps(checks, indent=2))

    for row in metrics.to_dict("records"):
        logger.log_metrics(
            {
                f"{row['split']}/{row['variant']}/{key}": float(value)
                for key, value in row.items()
                if key not in ("variant", "split") and np.isfinite(float(value))
            }
        )
    for row in overlap[overlap["stable"]].to_dict("records"):
        logger.log_metrics(
            {
                f"test/concept_{row['letter']}/{key}": float(row[key])
                for key in (
                    "spearman_kappa_refit_test",
                    "top_overlap_pct",
                    "top_capped_overlap_pct",
                )
            }
        )
    for row in shares.query("split == 'test'").to_dict("records"):
        if row["letter"] in stable:
            logger.log_metrics(
                {
                    f"test/concept_{row['letter']}/share_abs_phi_l1": row[
                        "share_abs_phi_l1"
                    ],
                    f"test/concept_{row['letter']}/share_abs_phi_refit": row[
                        "share_abs_phi_refit"
                    ],
                }
            )
    for name in ("metrics.csv", "importance.csv", "checks.json", "overlap.csv"):
        logger.log_artifact(str(out / name))
    for source in (__file__, "configs/xai/refit_magnitudes.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")


if __name__ == "__main__":
    main()
