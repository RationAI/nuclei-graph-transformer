"""Trains sparse autoencoders (and PCA baselines) on all validation nucleus representations.

Holdout: `holdout_size` of the validation slides, stratified by Gleason score, drawn
once with `seed` and shared by every k, seed and λ (`sae/holdout_slides.csv`).

Training: minibatches are drawn with replacement so that carcinoma and benign nuclei
are balanced 1:1 and no slide contributes more than about `max_slide_share` of the
samples; early stopping uses the loss on the holdout under the same weighting.

λ selection, per k and seed, on the holdout slides only (unweighted, i.e. the natural
distribution as in the test evaluation): among the λ of the grid with L0 within
`selection.l0_range` and dead fraction ≤ `selection.max_dead_fraction`, the one with
the lowest holdout RE. If no λ qualifies, nothing is selected for that (k, seed) and
this is reported; the conditions are not relaxed. The test set is never used.

PCA baselines with the same k are fitted on the same (non-holdout) validation
nuclei with the same sampling weights. Outputs: `sae/*.pt`, `sae/lambda_grid.csv`
(every trained model and its holdout metrics), `sae/models.csv` (selected models).
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.model_selection import train_test_split
from torch import Tensor

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import Activations, load_activations
from nuclei_graph.xai.runtime import setup_precision
from nuclei_graph.xai.sae import (
    ConceptAutoencoder,
    PCAAutoencoder,
    encode,
    load_autoencoder,
    save_autoencoder,
    train_sae,
)
from nuclei_graph.xai.sampling import balanced_slide_capped_weights


def holdout_split(slides: pd.DataFrame, size: float, seed: int) -> pd.Series:
    """Holdout validation slide ids, stratified by Gleason score."""
    validation = slides[slides["split"] == "validation"].reset_index(drop=True)
    _, holdout = train_test_split(
        validation,
        test_size=size,
        random_state=seed,
        stratify=validation["gleason_score"],
    )
    return holdout["slide_id"].sort_values().reset_index(drop=True)


@torch.no_grad()
def holdout_metrics(
    model: ConceptAutoencoder, z: Tensor, weights: np.ndarray
) -> dict[str, float]:
    """L0, dead fraction and RE on the holdout, unweighted (and L0 weighted)."""
    codes = encode(model, z)
    z_hat = model.cpu().decode(codes)
    sq_err = (z.float() - z_hat).pow(2).sum(1).double().numpy()
    active = (codes != 0).numpy()
    return {
        "L0": float(active.sum(1).mean()),
        "dead_fraction": float((~active.any(0)).mean()),
        "RE": float(np.sqrt(sq_err.mean())),
        "L0_weighted": float(active.sum(1) @ weights / weights.sum()),
        "RE_weighted": float(np.sqrt(sq_err @ weights / weights.sum())),
    }


def select(grid: pd.DataFrame, selection: DictConfig) -> pd.DataFrame:
    """Per (k, seed): the qualifying λ with the lowest holdout RE."""
    lo, hi = selection.l0_range
    grid["qualifies"] = grid["L0"].between(lo, hi) & (
        grid["dead_fraction"] <= selection.max_dead_fraction
    )
    grid["selected"] = False
    for _, group in grid.groupby(["k", "seed"]):
        qualifying = group[group["qualifies"]]
        if len(qualifying):
            grid.loc[qualifying["RE"].idxmin(), "selected"] = True
    return grid


def load_part(
    validation: Activations, rows: np.ndarray, max_slide_share: float, device: str
) -> tuple[Tensor, np.ndarray, dict[str, float]]:
    weights, stats = balanced_slide_capped_weights(
        validation.labels[rows], validation.slide_codes[rows], max_slide_share
    )
    z = torch.from_numpy(np.ascontiguousarray(validation.h[rows])).to(device)  # float16
    return z, weights, stats


@with_cli_args(["+xai=train_sae"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    for key, value in setup_precision(config.matmul_precision).items():
        logger.experiment.set_tag(logger.run_id, key, value)
    torch.use_deterministic_algorithms(True, warn_only=True)
    sae_dir = Path(config.sae_dir)
    sae_dir.mkdir(parents=True, exist_ok=True)

    slides = pd.read_parquet(config.slides)
    holdout_ids = holdout_split(slides, config.holdout_size, config.seed)
    holdout_path = sae_dir / "holdout_slides.csv"
    if holdout_path.exists():  # reused models must share the holdout
        previous = pd.read_csv(holdout_path)["slide_id"]
        assert previous.tolist() == holdout_ids.tolist(), "Holdout slides changed"
    holdout_ids.to_csv(holdout_path, index=False)

    validation = load_activations(config.activations_dir, "validation")
    is_holdout = (
        validation.index["slide_id"].astype(str).isin(set(holdout_ids)).to_numpy()
    )
    z_fit, w_fit, stats_fit = load_part(
        validation,
        np.flatnonzero(~is_holdout),
        config.sampler.max_slide_share,
        config.training.device,
    )
    z_holdout, w_holdout, stats_holdout = load_part(
        validation, np.flatnonzero(is_holdout), config.sampler.max_slide_share, "cpu"
    )
    for part, stats, n in (
        ("fit", stats_fit, len(z_fit)),
        ("holdout", stats_holdout, len(z_holdout)),
    ):
        print(f"{part}: {n:,} nuclei, sampler {stats}")
        logger.log_metrics({f"sampler/{part}/{k}": v for k, v in stats.items()})
        logger.log_metrics({f"sampler/{part}/nuclei": n})
    logger.log_metrics({"holdout/slides": len(holdout_ids)})

    rows = []
    for num_concepts in config.num_concepts:
        for seed in config.seeds:
            for l1_coeff in sorted(float(lam) for lam in config.l1_coeff.grid):
                path = sae_dir / f"sae_k{num_concepts}_seed{seed}_l1{l1_coeff:g}.pt"
                if config.reuse_existing and path.exists():
                    # trained by an earlier run of this grid with the same settings
                    model, stored = load_autoencoder(str(path))
                    assert (stored["k"], stored["seed"], stored["l1_coeff"]) == (
                        num_concepts,
                        seed,
                        l1_coeff,
                    ), f"Unexpected model in {path}"
                    metrics = holdout_metrics(model, z_holdout, w_holdout)
                    meta = {
                        key: stored[key]
                        for key in ("method", "k", "seed", "l1_coeff", "best_epoch")
                    }
                    meta |= {
                        "epochs": stored["epochs"],
                        "holdout_loss": stored["holdout_loss"],
                        **metrics,
                        "reused": True,
                    }
                else:
                    result = train_sae(
                        z_fit,
                        w_fit,
                        z_holdout,
                        w_holdout,
                        num_concepts=num_concepts,
                        l1_coeff=l1_coeff,
                        seed=seed,
                        **config.training,
                    )
                    metrics = holdout_metrics(result.model, z_holdout, w_holdout)
                    meta = {
                        "method": "sae",
                        "k": num_concepts,
                        "seed": seed,
                        "l1_coeff": l1_coeff,
                        "best_epoch": result.best_epoch,
                        "epochs": len(result.history),
                        "holdout_loss": result.best_val_loss,
                        **metrics,
                        "reused": False,
                    }
                    save_autoencoder(
                        result.model,
                        str(path),
                        **meta,
                        training=dict(config.training),
                        history=result.history,
                    )
                rows.append({**meta, "path": path.name})
                print(
                    f"k={num_concepts} seed={seed} λ={l1_coeff:g}"
                    f"{' (reused)' if meta['reused'] else ''}: "
                    f"L0={metrics['L0']:.2f} dead={metrics['dead_fraction']:.2f} "
                    f"RE={metrics['RE']:.3f} epochs={meta['epochs']}"
                )

        pca = PCAAutoencoder(validation.h.shape[1], num_concepts).fit(z_fit, w_fit)
        path = sae_dir / f"pca_k{num_concepts}.pt"
        meta = {
            "method": "pca",
            "k": num_concepts,
            "seed": 0,
            **holdout_metrics(pca, z_holdout, w_holdout),
        }
        save_autoencoder(pca, str(path), **meta)
        rows.append({**meta, "path": path.name})

    grid = pd.DataFrame(rows)
    sae = select(grid[grid["method"] == "sae"].copy(), config.selection)
    sae.to_csv(sae_dir / "lambda_grid.csv", index=False)
    selected = pd.concat(
        [sae[sae["selected"]], grid[grid["method"] == "pca"]], ignore_index=True
    )
    selected.to_csv(sae_dir / "models.csv", index=False)

    missing = sorted(
        set(map(tuple, sae[["k", "seed"]].drop_duplicates().to_numpy()))
        - set(map(tuple, sae.loc[sae["selected"], ["k", "seed"]].to_numpy()))
    )
    for k, seed in missing:
        print(f"[WARN] k={k} seed={seed}: no λ satisfies the selection conditions")
    for row in sae[sae["selected"]].itertuples(index=False):
        logger.log_metrics(
            {
                f"selected/k{row.k}/seed{row.seed}/{key}": float(getattr(row, key))
                for key in ("l1_coeff", "L0", "dead_fraction", "RE")
            }
        )
    logger.log_metrics({"selection/unselected_k_seed": len(missing)})
    logger.log_artifacts(str(sae_dir), "sae")


if __name__ == "__main__":
    main()
