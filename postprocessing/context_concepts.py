# ruff: noqa: RUF002  (mathematical symbols in text are intended)
r"""fig:res-context-concepts: on which nuclei the context helps, by dominating SAE concept.

PANDA Radboud test set, all 14.7 M test nuclei. Stored outputs only: the test predictions of the
three context configurations of tab:res-context (`postprocessing.results_context`, run 027eeaee:
self-only, k-NN with k = 24 and dense, shape embedding without position encoding, three seeds
each) and the stored codes of the analysed SAE (k = 16, L1, seed 1; SAE training run 4cf1db8f)
of the thesis model c8f26853. No training, no inference.

Grouping. φ_ij = κ_ij wᵀu_j, the exact contribution of concept j to the SAE's reconstructed logit
(stored as `contrib_test.npy`; checked against the stored codes κ times wᵀu_j from the SAE
decoder and the class head). A nucleus belongs to the stable concept (A-H, by the deck key)
with the largest |φ_ij|; nuclei dominated by an unstable concept, or with no active concept
(κ_i = 0), form the group "other". The XAI rows are matched to the predictions by (slide,
64-bit nucleus key); the stored label must equal the evaluation label on every nucleus.

Per group and configuration: the mean log-loss of the seed-averaged probability
p̄_i = mean over the three seeds of p_i (clipped to [ε, 1 − ε]). Differences k-NN − self-only and
dense − k-NN of the group means (negative = the larger context lowers the loss), with percentile
CIs from the stratified slide bootstrap of the thesis (the resamples of tab:res-context, seed
20261108, B = 2,000): in each resample a group mean is the multiplicity-weighted sum of the
slides' loss sums over the weighted sum of their nucleus counts. p = 2 x the smaller tail
fraction, unadjusted (the figure is descriptive). Also for all nuclei together.

Outputs in `out_dir` (no slide ids): `groups.csv` (one row per group and "all"),
`replicates.npz`, `checks.json`, `figures/res_context_concepts.{pdf,png}`.
"""

import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.metrics import LinearHead, concept_logit_weights
from nuclei_graph.xai.sae import load_autoencoder
from postprocessing.bootstrap_sources import (
    bootstrap_p,
    label_predictions,
    load_labels,
    load_run,
    nucleus_keys,
)
from postprocessing.results_context import REPO, load_slides
from visualization.context_concepts_figure import save_context_concepts_figure
from xai.concept_morphometry import concept_letters


OTHER = "other"
CHUNK = 1_000_000
SOURCES = (
    "postprocessing/context_concepts.py",
    "postprocessing/results_context.py",
    "postprocessing/bootstrap_sources.py",
    "visualization/context_concepts_figure.py",
    "configs/postprocessing/context_concepts.yaml",
    "configs/postprocessing/results_context.yaml",
)


def code_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *SOURCES)
    return f"{git('rev-parse', 'HEAD')}{' (dirty)' if dirty else ''}"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def seed_averaged_predictions(
    config: DictConfig, slide_ids: list[str], labels: pd.DataFrame, carcinoma: NDArray
) -> tuple[pd.DataFrame, NDArray, dict[str, NDArray]]:
    """(slide, key) of the nuclei, their labels and each configuration's mean probability."""
    reference, y, p_mean = None, None, {}
    for name in config.configurations_used:
        total = None
        for run_id in config.configurations[name].eval_runs:
            df, _ = load_run(
                run_id, slide_ids, Path(config.cache_dir), config.predictions_artifact
            )
            if reference is None:
                reference = df[["slide", "key"]]
                y, _ = label_predictions(df, labels, carcinoma)
            assert np.array_equal(df["slide"].to_numpy(), reference["slide"].to_numpy())
            assert np.array_equal(df["key"].to_numpy(), reference["key"].to_numpy())
            p = df["p"].to_numpy(np.float64)
            total = p if total is None else total + p
            print(f"{name} {run_id[:8]}: loaded", flush=True)
        p_mean[name] = total / len(config.configurations[name].eval_runs)
    assert reference is not None and y is not None
    return reference, y, p_mean


def xai_order(
    config: DictConfig, slide_ids: list[str], nuclei: pd.DataFrame, y: NDArray
) -> NDArray:
    """Row order of the XAI arrays that matches the predictions' (slide, key) order."""
    index = pd.read_parquet(
        config.xai.index, columns=["slide_id", "nucleus_id", "label"]
    )
    code = pd.Categorical(index["slide_id"].astype(str), categories=slide_ids).codes
    assert (code >= 0).all(), "XAI rows outside the test slides"
    keys = nucleus_keys(index["nucleus_id"])
    order = np.lexsort((keys, code))
    assert np.array_equal(code[order], nuclei["slide"].to_numpy()), "slides differ"
    assert np.array_equal(keys[order], nuclei["key"].to_numpy()), "nuclei differ"
    assert np.array_equal(index["label"].to_numpy()[order], y), "labels differ"
    return order


def dominant_groups(
    config: DictConfig, order: NDArray
) -> tuple[NDArray, list[str], dict]:
    """Group code per nucleus (stable letters in alphabetical order, then "other")."""
    x = config.xai
    model, meta = load_autoencoder(x.sae_model)
    assert (meta["k"], meta["seed"]) == (16, 1), meta
    head = LinearHead.from_npz(str(x.head))
    weights = concept_logit_weights(
        model.decoder_weight.detach().double().numpy(), head
    )

    letters = dict(concept_letters(x.deck_key))
    stable = sorted(j for j, letter in letters.items() if letter in x.stable_concepts)
    assert stable == sorted(x.stable_indices), (stable, x.stable_indices)
    group_names = [*sorted(letters[j] for j in stable), OTHER]
    group_of_index = np.full(len(weights), len(group_names) - 1, np.int8)
    for j in stable:
        group_of_index[j] = group_names.index(letters[j])

    phi_all = np.load(x.contributions, mmap_mode="r")
    codes_all = np.load(x.codes, mmap_mode="r")
    n = len(order)
    assert phi_all.shape == codes_all.shape == (n, len(weights))
    group = np.empty(n, np.int8)
    dominant = np.empty(n, np.int8)
    inactive = np.zeros(n, bool)
    max_dev = 0.0
    for start in range(0, n, CHUNK):
        rows = order[start : start + CHUNK]
        phi = np.asarray(phi_all[rows], np.float64)
        kappa = np.asarray(codes_all[rows], np.float64)
        max_dev = max(max_dev, float(np.abs(phi - kappa * weights).max()))
        j = np.abs(phi).argmax(1)
        dominant[start : start + CHUNK] = j
        inactive[start : start + CHUNK] = ~(kappa > 0).any(1)
        group[start : start + CHUNK] = group_of_index[j]
    group[inactive] = len(group_names) - 1
    assert max_dev <= x.contribution_tolerance, f"stored φ deviates by {max_dev}"
    unstable = ~np.isin(dominant, stable) & ~inactive
    checks = {
        "contribution_max_abs_deviation": max_dev,
        "concept_logit_weights": {
            letters[j] if j in stable else f"#{j}": float(weights[j])
            for j in range(len(weights))
        },
        "other_inactive": int(inactive.sum()),
        "other_unstable_dominant": int(unstable.sum()),
        "other_unstable_by_index": {
            f"#{j}": int((unstable & (dominant == j)).sum())
            for j in range(len(weights))
            if j not in stable
        },
    }
    return group, group_names, checks


# --------------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------------


def log_loss(y: NDArray, p: NDArray, eps: float) -> NDArray:
    p = np.clip(p, eps, 1 - eps)
    return -np.where(y == 1, np.log(p), np.log1p(-p))


def load_grouped(config: DictConfig) -> dict:
    """Test nuclei with labels, seed-mean probabilities and dominating-concept groups."""
    slides = load_slides(config.metadata_uri)
    slide_ids = slides["slide_id"].tolist()
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    nuclei, y, p_mean = seed_averaged_predictions(config, slide_ids, labels, carcinoma)
    order = xai_order(config, slide_ids, nuclei, y)
    group, group_names, group_checks = dominant_groups(config, order)
    return {
        "slide_ids": slide_ids,
        "slide": nuclei["slide"].to_numpy().astype(np.int64),
        "y": y,
        "p_mean": p_mean,
        "group": group,
        "group_names": group_names,
        "group_checks": group_checks,
    }


def group_table(
    config: DictConfig, data: dict
) -> tuple[pd.DataFrame, dict[str, NDArray], dict]:
    """Per group (and "all"): mean log-losses and the configured differences with CIs."""
    slide_ids, slide, y = data["slide_ids"], data["slide"], data["y"]
    p_mean, group, group_names = data["p_mean"], data["group"], data["group_names"]
    n_slides = len(slide_ids)
    n_groups = len(group_names)

    eps = float(config.probability_clip)
    clipped = {n: int(((p < eps) | (p > 1 - eps)).sum()) for n, p in p_mean.items()}
    cell = slide * n_groups + group
    size = n_slides * n_groups
    counts = (
        np.bincount(cell, minlength=size).reshape(n_slides, n_groups).astype(np.float64)
    )
    sums = {
        n: np.bincount(cell, weights=log_loss(y, p, eps), minlength=size).reshape(
            n_slides, n_groups
        )
        for n, p in p_mean.items()
    }
    carcinoma_n = np.bincount(group, weights=(y == 1), minlength=n_groups)

    saved = np.load(config.resamples)
    assert saved["slide_ids"].tolist() == slide_ids, "resamples: different slides"
    positive = np.bincount(slide, weights=(y == 1), minlength=n_slides) > 0
    assert np.array_equal(saved["positive"], positive), "resamples: different strata"
    weights = saved["multiplicities"].astype(np.float64)

    # columns: the groups, then all nuclei
    def with_all(a: NDArray) -> NDArray:
        return np.concatenate([a, a.sum(-1, keepdims=True)], -1)

    point = {n: with_all(s.sum(0)) / with_all(counts.sum(0)) for n, s in sums.items()}
    rep_counts = with_all(weights @ counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        rep = {n: with_all(weights @ s) / rep_counts for n, s in sums.items()}
    empty = int((rep_counts == 0).sum())

    pct = (1 - config.bootstrap.ci_level) / 2 * 100
    names = [*group_names, "all"]
    total = with_all(counts.sum(0))
    rows = []
    diff_reps = {}
    for g, name in enumerate(names):
        row = {
            "group": name,
            "n_nuclei": int(total[g]),
            "share_of_nuclei": float(total[g] / total[-1]),
            "n_carcinoma": int(carcinoma_n[g] if g < n_groups else carcinoma_n.sum()),
        }
        row["carcinoma_share"] = row["n_carcinoma"] / row["n_nuclei"]
        row |= {f"log_loss_{n}": float(point[n][g]) for n in config.configurations_used}
        for d in config.differences:
            reps = rep[d.plus][:, g] - rep[d.minus][:, g]
            diff_reps[f"{d.name}/{name}"] = reps
            lo, hi = np.nanpercentile(reps, [pct, 100 - pct])
            row |= {
                f"{d.name}": float(point[d.plus][g] - point[d.minus][g]),
                f"{d.name}_ci_lo": float(lo),
                f"{d.name}_ci_hi": float(hi),
                f"{d.name}_p": bootstrap_p(reps[~np.isnan(reps)]),
            }
        rows.append(row)

    checks = {
        "nuclei": len(y),
        "slides": n_slides,
        "rows_matched": "XAI index rows matched to the predictions by (slide, nucleus key); "
        "labels identical on every nucleus",
        "groups": group_names,
        **data["group_checks"],
        "probability_clip": eps,
        "clipped_probabilities": clipped,
        "resamples": {
            "file": str(config.resamples),
            "n_resamples": len(weights),
            "seed": int(saved["seed"]),
            "replicate_group_cells_without_nuclei": empty,
        },
    }
    return pd.DataFrame(rows), diff_reps, checks


def compute(config: DictConfig) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    table, diff_reps, checks = group_table(config, load_grouped(config))
    table.to_csv(out / "groups.csv", index=False)
    np.savez_compressed(out / "replicates.npz", **diff_reps)  # type: ignore[arg-type]

    figure_dirs = [out / "figures"]
    if config.get("repo_figures_dir"):
        figure_dirs.append(Path(config.repo_figures_dir))
    save_context_concepts_figure(out, figure_dirs)

    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    return out, checks


@with_cli_args(["+postprocessing=context_concepts"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    for key, value in config.links.items():
        client.set_tag(logger.run_id, key, value)
    out, _ = compute(config)
    table = pd.read_csv(out / "groups.csv")
    metrics = {}
    for r in table.to_dict("records"):
        g = r["group"]
        metrics[f"{g}/n_nuclei"] = float(r["n_nuclei"])
        metrics[f"{g}/carcinoma_share"] = float(r["carcinoma_share"])
        for n in config.configurations_used:
            metrics[f"{g}/log_loss_{n}"] = r[f"log_loss_{n}"]
        for d in config.differences:
            for suffix in ("", "_ci_lo", "_ci_hi"):
                metrics[f"{g}/{d.name}{suffix}"] = r[f"{d.name}{suffix}"]
    logger.log_metrics(metrics)
    for name in ("groups.csv", "checks.json", "replicates.npz"):
        logger.log_artifact(str(out / name))
    for name in ("res_context_concepts.pdf", "res_context_concepts.png"):
        logger.log_artifact(str(out / "figures" / name), "figures")
    for source in SOURCES:
        logger.log_artifact(str(REPO / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


if __name__ == "__main__":
    main()
