r"""Concepts across training seeds, SAE seed 1 against SAE seed 1, all latents, all test nuclei.

The analysed SAE (model c8f26853, k = 16, SAE seed 1, training run 4cf1db8f, stable concepts
A-H of tab:res-concepts) is compared with the SAE of the same seed, k, λ grid and selection
rule trained on the final representation h of each other training seed of the model
(`xai.train_sae` in the model's own folder; the seed-1 models of the earlier analysis
`xai.cross_model_concepts` are reused, they were trained with the code and config of
4cf1db8f).

Per model, on all PANDA Radboud test nuclei (tab:res-faithfulness, ℓ1 codes, no refit):
E_h = RE (RMS ‖h - ĥ‖), EV (1 - RE² / total variance of h), E_z = FE (RMS of f - f_A,
f = wᵀh + β, f_A = wᵀĥ + β), E_lin = MCE (OLS head on the validation codes), Spearman of f
with f_A, SD ratio std(f_A) / std(f), AUPRC of f_A, and the flip rate at the thesis Youden
threshold (p ≈ 0.40, the seed-mean validation threshold of `thresholds.json`, shared by all
seeds): (f ≥ t) ≠ (f_A ≥ t). The metric code is that of `xai.refit_magnitudes` and
`xai.faithfulness_flips`; the reference row is recomputed from the stored codes and checked
against the stored table.

Matching: Pearson correlation of the activations of every reference latent with every
latent of the new SAE over all test nuclei (accumulated in float64); one-to-one Hungarian
assignment on the 16 x 16 correlation. Decoder directions are not compared across models
(the representation spaces are not aligned). Per reference concept: the Hungarian partner
and its r, the best latent and its r (and the reference concept that latent is assigned to
or correlates with most), the second-best r, the multiple correlation R of the concept's
activation with all 16 new latents (OLS, a hint of a concept split over several latents),
the sign of wᵀu of the partner in its own model, its share of Σ_j mean|φ_j| (φ_j = wᵀu_j
c_j) and its SAE-seed stability in its own model (Hungarian decoder cosine to SAE seeds 0
and 2 of the same model; decoders are comparable there). Matched: r ≥ `min_correlation`
and the same sign.

Comparison with the earlier analysis (run d2ad948d, the SAE seed with the lowest holdout RE
= seed 0 in both models, 2M random test nuclei): its stored seed-0 codes are matched again
on the 2M nuclei (reproduces its table) and on all nuclei, and every seed-1 partner is
linked to its seed-0 counterpart in the same model by decoder cosine, which separates the
effect of the SAE seed from that of the nuclei.

    python -m xai.cross_model_seed1 +experiment=xai/radboud reference.dir=<analysed dir> \
        "models=[{name, run_id, dir, sae_training_run_id}, ...]"
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
from scipy.optimize import linear_sum_assignment

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import LinearHead, concept_logit_weights
from nuclei_graph.xai.runtime import setup_precision
from nuclei_graph.xai.sae import covariance, encode, load_autoencoder
from nuclei_graph.xai.stability import matched_cosines
from xai.concept_masks import code_commit
from xai.concept_morphometry import concept_letters
from xai.cross_model_concepts import contribution_shares, pearson_matrix
from xai.evaluate_sae import head_logits
from xai.faithfulness_flips import flip_rows, logit
from xai.refit_magnitudes import evaluate
from xai.train_sae import select


SPLITS = ("validation", "test")
# thesis symbol -> column of xai.refit_magnitudes.evaluate
FAITHFULNESS = {
    "E_h": "RE",
    "EV": "explained_variance",
    "E_z": "FE",
    "E_lin": "MCE",
    "Spearman": "spearman_f_fA",
    "SD_ratio": "std_ratio_fA_f",
    "AUPRC_fA": "AUPRC_fA",
}


# --- SAE of one model ---


def selected_sae(sae_dir: Path, k: int, seed: int, selection: DictConfig) -> dict:
    """The selected λ of (k, seed) from models.csv, re-derived from lambda_grid.csv."""
    models = pd.read_csv(sae_dir / "models.csv")
    row = models[
        (models["method"] == "sae") & (models["k"] == k) & (models["seed"] == seed)
    ]
    assert len(row) == 1, f"{sae_dir}: {len(row)} selected SAEs for k={k} seed={seed}"
    row = row.iloc[0].to_dict()
    grid = pd.read_csv(sae_dir / "lambda_grid.csv")
    grid = grid[(grid["k"] == k) & (grid["seed"] == seed)].copy()
    assert len(grid) == 13, f"{sae_dir}: λ grid of {len(grid)} values"
    redone = select(grid.drop(columns=["qualifies", "selected"]), selection)
    assert redone.loc[redone["selected"], "l1_coeff"].tolist() == [row["l1_coeff"]]
    row["qualifying_l1"] = redone.loc[redone["qualifies"], "l1_coeff"].tolist()
    return row


def seed_stability(sae_dir: Path, k: int, seed: int, threshold: float) -> pd.DataFrame:
    """Hungarian decoder cosine of every latent of `seed` to the other selected seeds."""
    models = pd.read_csv(sae_dir / "models.csv")
    models = models[(models["method"] == "sae") & (models["k"] == k)]
    paths = dict(zip(models["seed"].astype(int), models["path"], strict=True))
    decoder = {
        s: load_autoencoder(str(sae_dir / p))[0].decoder_weight.detach().numpy()
        for s, p in paths.items()
    }
    out = pd.DataFrame({"latent": np.arange(k)})
    for other in sorted(paths):
        if other != seed:
            match, cosine = matched_cosines(decoder[seed], decoder[other])
            out[f"seed{other}_latent"] = match
            out[f"seed{other}_cosine"] = cosine
    out["min_cosine"] = out.filter(like="_cosine").min(1)
    out["stable"] = out["min_cosine"] >= threshold
    return out


def model_evaluation(
    model_dir: Path,
    sae_path: Path,
    stored_codes: dict[str, Path] | None,
    t_logit: float,
    config: DictConfig,
) -> tuple[dict, NDArray, NDArray, NDArray, NDArray, dict]:
    """tab:res-faithfulness on test, the test codes, wᵀu, |φ| shares, labels and checks."""
    model, meta = load_autoencoder(str(sae_path))
    assert (meta["k"], meta["seed"]) == (config.num_concepts, config.sae_seed), meta
    decoder = model.decoder_weight.detach().double().numpy()
    b_d = model.decoder_bias.detach().double().numpy()
    head = LinearHead.from_npz(str(model_dir / "activations" / "head.npz"))
    data = {s: load_activations(model_dir / "activations", s) for s in SPLITS}
    f = {s: head_logits(head, a) for s, a in data.items()}
    total_variance = {s: float(np.trace(covariance(a.h)[1])) for s, a in data.items()}
    checks: dict[str, float] = {}

    device = config.device
    model = model.to(device)
    codes = {}
    for split in SPLITS:
        if stored_codes is None:
            codes[split] = encode(model, data[split].h, device=device).numpy()
        else:
            codes[split] = np.load(stored_codes[split])
            head_rows = slice(0, 1_000_000)
            fresh = encode(model, data[split].h[head_rows], device=device).numpy()
            checks[f"{split}/encoding_max_abs_diff_vs_stored_first_1M"] = float(
                np.abs(fresh - codes[split][head_rows]).max()
            )
        assert codes[split].shape == (len(data[split]), config.num_concepts)
    model = model.cpu()

    rows = evaluate(
        codes,
        data,
        f,
        decoder,
        b_d,
        head,
        total_variance,
        config.flip_margin,
        config.chunk_size,
    )
    test_row = next(r for r in rows if r["split"] == "test")
    test = data["test"]
    f_a = np.concatenate(
        [
            head(np.asarray(codes["test"][r], np.float64) @ decoder + b_d)
            for r in test.chunks(config.chunk_size)
        ]
    )
    flips = flip_rows(f["test"], f_a, test.labels, t_logit, strict=False)
    overall = next(r for r in flips if r["band"] == "all" and r["label"] == "all")
    row = {symbol: test_row[column] for symbol, column in FAITHFULNESS.items()}
    row |= {
        "flip_rate_youden": overall["flip_rate"],
        "flip_rate_zero": test_row["flip_rate"],
        "AUPRC_f": test_row["AUPRC_f"],
        "AUROC_fA": test_row["AUROC_fA"],
        "L0_test": test_row["L0"],
        "dead_fraction_test": test_row["dead_fraction"],
    }
    weights = concept_logit_weights(decoder, head)
    shares = contribution_shares(codes["test"], weights)
    return row, codes["test"], weights, shares, test.labels, checks


# --- matching ---


def cross_statistics(
    a: NDArray, b: NDArray, chunk: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Pearson r (k_a x k_b), cov(a, b), cov(b) and var(a), float64 over all rows."""
    ka = a.shape[1]
    total = np.zeros(ka + b.shape[1])
    gram = np.zeros((len(total), len(total)))
    for start in range(0, len(a), chunk):
        x = np.hstack(
            [
                np.asarray(a[start : start + chunk], np.float64),
                np.asarray(b[start : start + chunk], np.float64),
            ]
        )
        total += x.sum(0)
        gram += x.T @ x
    mean = total / len(a)
    cov = gram / len(a) - np.outer(mean, mean)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov[:ka, ka:] / np.outer(sd[:ka], sd[ka:])
    return np.nan_to_num(r), cov[:ka, ka:], cov[ka:, ka:], np.diag(cov)[:ka]


def multiple_correlation(cov_ab: NDArray, cov_bb: NDArray, var_a: NDArray) -> NDArray:
    """R of each column of a regressed (OLS, with intercept) on all columns of b."""
    explained = np.einsum("ij,jk,ik->i", cov_ab, np.linalg.pinv(cov_bb), cov_ab)
    return np.sqrt(np.clip(explained / var_a, 0, 1))


def match_table(
    corr: NDArray,
    multiple_r: NDArray,
    letters: dict[int, str],
    stable: list[str],
    ref_w: NDArray,
    w: NDArray,
    shares: NDArray,
    stability: pd.DataFrame,
    min_correlation: float,
) -> tuple[pd.DataFrame, dict[int, int]]:
    rows_h, cols_h = linear_sum_assignment(-corr)
    hungarian = dict(zip(rows_h.tolist(), cols_h.tolist(), strict=True))
    assigned_to = {latent: concept for concept, latent in hungarian.items()}
    index = {letter: j for j, letter in letters.items()}
    rows = []
    for letter in stable:
        j = index[letter]
        partner = hungarian[j]
        best = int(np.argmax(corr[j]))
        best_owner = assigned_to[best]
        best_top = int(np.argmax(corr[:, best]))  # reference concept with the highest r
        same_sign = bool(np.sign(ref_w[j]) == np.sign(w[partner]))
        rows.append(
            {
                "letter": letter,
                "partner": partner,
                "r": float(corr[j, partner]),
                "best_latent": best,
                "best_r": float(corr[j, best]),
                "best_is_partner": best == partner,
                "best_assigned_to": letters[best_owner],
                "best_correlates_most_with": letters[best_top],
                "best_r_with_that": float(corr[best_top, best]),
                "second_r": float(np.sort(corr[j])[-2]),
                "multiple_R": float(multiple_r[j]),
                "logit_weight_reference": float(ref_w[j]),
                "logit_weight_partner": float(w[partner]),
                "same_sign": same_sign,
                "partner_share_abs_phi": float(shares[partner]),
                "partner_seed_stability": float(stability.loc[partner, "min_cosine"]),
                "partner_stable": bool(stability.loc[partner, "stable"]),
                "matched": bool(corr[j, partner] >= min_correlation and same_sign),
            }
        )
    return pd.DataFrame(rows), hungarian


def previous_comparison(
    name: str,
    model_dir: Path,
    ref_codes: NDArray,
    letters: dict[int, str],
    stable: list[str],
    ref_w: NDArray,
    seed1_table: pd.DataFrame,
    config: DictConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """The earlier seed-0 matching (d2ad948d) redone on its 2M nuclei and on all nuclei."""
    prev_dir = Path(config.previous.dir)
    previous = pd.read_csv(prev_dir / "matching.csv")
    previous = previous[previous["model"] == name].set_index("letter")
    concepts = pd.read_csv(prev_dir / "concepts.csv")
    prev_seed = int(concepts.loc[concepts["model"] == name, "seed"].iloc[0])
    codes0 = np.load(prev_dir / f"codes_test_{name}.npy", mmap_mode="r")
    sae_dir = model_dir / "sae"
    paths = {
        s: sae_dir / f"sae_k{config.num_concepts}_seed{s}_l1"
        f"{selected_sae(sae_dir, config.num_concepts, s, config.selection)['l1_coeff']:g}.pt"
        for s in (prev_seed, config.sae_seed)
    }
    decoders = {
        s: load_autoencoder(str(p))[0].decoder_weight.detach().numpy()
        for s, p in paths.items()
    }
    # seed-1 latent -> its seed-0 counterpart (Hungarian decoder cosine, same model)
    counterpart, cosine = matched_cosines(
        decoders[config.sae_seed], decoders[prev_seed]
    )
    w0 = concept_logit_weights(
        decoders[prev_seed].astype(np.float64),
        LinearHead.from_npz(str(model_dir / "activations" / "head.npz")),
    )

    n = len(ref_codes)
    sub = np.sort(
        np.random.default_rng(config.seed).choice(
            n, config.previous.subsample, replace=False
        )
    )
    corr_sub = pearson_matrix(ref_codes[sub], codes0[sub])
    corr_all = cross_statistics(ref_codes, codes0, config.chunk_size)[0]
    hung_sub = dict(zip(*linear_sum_assignment(-corr_sub), strict=True))
    hung_all = dict(zip(*linear_sum_assignment(-corr_all), strict=True))
    index = {letter: j for j, letter in letters.items()}
    rows, reproduced = [], []
    seed1 = seed1_table.set_index("letter")
    for letter in stable:
        j = index[letter]
        p_sub, p_all = int(hung_sub[j]), int(hung_all[j])
        p1 = int(seed1.loc[letter, "partner"])
        reproduced.append(
            abs(corr_sub[j, p_sub] - previous.loc[letter, "correlation"])
            + abs(p_sub - previous.loc[letter, "partner"])
        )
        matched_all = bool(
            corr_all[j, p_all] >= config.match.min_correlation
            and np.sign(w0[p_all]) == np.sign(ref_w[j])
        )
        rows.append(
            {
                "letter": letter,
                "previous_seed": prev_seed,
                "previous_partner": int(previous.loc[letter, "partner"]),
                "previous_r_2M": float(previous.loc[letter, "correlation"]),
                "previous_matched": bool(previous.loc[letter, "matched"]),
                "seed0_partner_all": p_all,
                "seed0_r_all": float(corr_all[j, p_all]),
                "seed0_matched_all": matched_all,
                "seed1_partner": p1,
                "seed1_r_all": float(seed1.loc[letter, "r"]),
                "seed1_matched": bool(seed1.loc[letter, "matched"]),
                "seed1_partner_seed0_counterpart": int(counterpart[p1]),
                "seed1_partner_seed0_cosine": float(cosine[p1]),
                "same_feature_as_previous": int(counterpart[p1]) == p_all,
                "verdict_changed": bool(seed1.loc[letter, "matched"])
                != bool(previous.loc[letter, "matched"]),
            }
        )
    checks = {f"{name}/previous_2M_max_abs_diff": float(max(reproduced))}
    return pd.DataFrame(rows), checks


# --- driver ---


@with_cli_args(["+xai=cross_model_seed1"])
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
    client.set_tag(
        logger.run_id, "xai.reference_sae_run_id", config.reference.sae_training_run_id
    )
    client.set_tag(logger.run_id, "xai.previous_run_id", config.previous.run_id)
    for entry in config.models:
        client.set_tag(
            logger.run_id, f"xai.sae_run_id.{entry.name}", entry.sae_training_run_id
        )
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    k, seed = config.num_concepts, config.sae_seed
    stable = list(config.stable_concepts)
    checks: dict[str, object] = {}

    youden = json.loads(Path(config.thresholds).read_text())["thresholds"]["youden"]
    t_logit = logit(float(youden["threshold"]))

    ref_dir = Path(config.reference.dir)
    letters = dict(concept_letters(str(ref_dir / config.reference.deck_key)))
    models = [
        {
            "name": "reference",
            "run_id": config.parent_run_id,
            "dir": ref_dir,
            "sae_training_run_id": config.reference.sae_training_run_id,
            "stored": {s: ref_dir / config.reference.codes[s] for s in SPLITS},
        }
    ] + [
        {**entry, "dir": Path(entry.dir), "stored": None} for entry in config.models
    ]

    selection, faithfulness, results = [], [], {}
    for m in models:
        sae_dir = m["dir"] / "sae"
        chosen = selected_sae(sae_dir, k, seed, config.selection)
        selection.append(
            {
                "model": m["name"],
                "training_run_id": m["run_id"],
                "sae_training_run_id": m["sae_training_run_id"],
                **{
                    c: chosen[c]
                    for c in ("l1_coeff", "L0", "dead_fraction", "RE", "best_epoch")
                },
                "qualifying_l1": chosen["qualifying_l1"],
            }
        )
        row, codes, w, shares, labels, model_checks = model_evaluation(
            m["dir"], sae_dir / chosen["path"], m["stored"], t_logit, config
        )
        checks |= {f"{m['name']}/{key}": v for key, v in model_checks.items()}
        faithfulness.append({"model": m["name"], **row})
        print(m["name"], json.dumps(faithfulness[-1]), flush=True)
        stability = seed_stability(sae_dir, k, seed, config.stability_threshold)
        results[m["name"]] = (codes, w, shares, stability, labels)
        if m["stored"] is None:
            np.save(out / f"codes_test_{m['name']}_seed{seed}.npy", codes)

    selection = pd.DataFrame(selection)
    faithfulness = pd.DataFrame(faithfulness)
    selection.to_csv(out / "sae_selection.csv", index=False)
    faithfulness.to_csv(out / "faithfulness.csv", index=False)

    # the reference row against the stored tab:res-faithfulness values
    stored = pd.read_csv(ref_dir / config.reference.metrics)
    stored = stored.query("variant == 'sae_l1' and split == 'test'").iloc[0]
    flips = pd.read_csv(config.reference_flips)
    stored_flip = flips.query(
        "threshold == 'youden' and band == 'all' and label == 'all'"
    )["sae_k16_seed1_l1"].iloc[0]
    ref_row = faithfulness.set_index("model").loc["reference"]
    diffs = {
        symbol: abs(float(ref_row[symbol]) - float(stored[column]))
        for symbol, column in FAITHFULNESS.items()
    }
    diffs["flip_rate_youden"] = abs(float(ref_row["flip_rate_youden"]) - stored_flip)
    checks["reference_vs_stored_table_abs_diff"] = diffs
    assert max(diffs.values()) < 1e-4, diffs

    ref_codes, ref_w, ref_shares, ref_stability, ref_labels = results["reference"]
    matches, correlations, latents, comparisons = [], [], [], []
    for m in models:
        name = m["name"]
        codes, w, shares, stability, labels = results[name]
        assert np.array_equal(labels, ref_labels), f"{name}: test nuclei differ"
        latents.append(
            stability.assign(
                model=name,
                logit_weight=w,
                sign=np.sign(w).astype(int),
                share_abs_phi=shares,
                activation_frequency=(codes > 0).mean(0),
            )
        )
        if name == "reference":
            continue
        corr, cov_ab, cov_bb, var_a = cross_statistics(
            ref_codes, codes, config.chunk_size
        )
        table, hungarian = match_table(
            corr,
            multiple_correlation(cov_ab, cov_bb, var_a),
            letters,
            stable,
            ref_w,
            w,
            shares,
            stability,
            config.match.min_correlation,
        )
        table.insert(0, "model", name)
        matches.append(table)
        correlations.append(
            pd.DataFrame(corr, columns=[f"latent{j}" for j in range(k)]).assign(
                model=name,
                reference_concept=[letters[j] for j in range(k)],
                hungarian_partner=[hungarian[j] for j in range(k)],
            )
        )
        comparison, comparison_checks = previous_comparison(
            name,
            m["dir"],
            ref_codes,
            letters,
            stable,
            ref_w,
            table,
            config,
        )
        checks |= comparison_checks
        comparisons.append(comparison.assign(model=name))
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(f"\n{name}\n{table.round(3).to_string(index=False)}")
            print(comparison.round(3).to_string(index=False))
        logger.log_metrics(
            {
                f"{name}/matched": int(table["matched"].sum()),
                f"{name}/matched_main_GBHA": int(
                    table.set_index("letter").loc[["G", "B", "H", "A"], "matched"].sum()
                ),
            }
        )

    matches = pd.concat(matches, ignore_index=True)
    matches.to_csv(out / "matching.csv", index=False)
    pd.concat(correlations, ignore_index=True).to_csv(
        out / "correlations.csv", index=False
    )
    pd.concat(latents, ignore_index=True).to_csv(out / "latents.csv", index=False)
    pd.concat(comparisons, ignore_index=True).to_csv(
        out / "previous_comparison.csv", index=False
    )
    count = matches.groupby("letter", sort=False)["matched"].sum()
    summary = pd.DataFrame(
        {
            "letter": count.index,
            "matched_in_models": count.to_numpy(),
            "of": matches["model"].nunique(),
        }
    )
    summary.to_csv(out / "summary.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print(summary.to_string(index=False))
    print(json.dumps(checks, indent=1))

    for row in faithfulness.to_dict("records"):
        logger.log_metrics(
            {
                f"{row['model']}/{key}": float(value)
                for key, value in row.items()
                if key != "model"
            }
        )
    for row in selection.to_dict("records"):
        logger.log_metrics(
            {
                f"{row['model']}/selected_{key}": float(row[key])
                for key in ("l1_coeff", "L0", "dead_fraction", "RE")
            }
        )
    logger.log_metrics({"matched_in_all": int((count == len(config.models)).sum())})
    for name in (
        "sae_selection.csv",
        "faithfulness.csv",
        "matching.csv",
        "correlations.csv",
        "latents.csv",
        "previous_comparison.csv",
        "summary.csv",
        "checks.json",
    ):
        logger.log_artifact(str(out / name), "cross_model_seed1")
    for source in (__file__, "configs/xai/cross_model_seed1.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")


if __name__ == "__main__":
    torch.set_num_threads(16)
    main()
