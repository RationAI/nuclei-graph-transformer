"""Trains overcomplete TopK SAEs (Gao et al. 2024) on the validation representations.

Data exactly as for the k = 16 SAE (`xai.train_sae`): the validation slides without the
40 holdout slides of `sae/holdout_slides.csv` (recomputed here and compared), minibatches
drawn with the same balanced, slide-capped sampler; the holdout slides are used for
early stopping (sampler-weighted MSE, as the k = 16 SAE) and reported unweighted. The
test set is not touched except for the data check of the stored k = 16 codes.

Grid: `grid.num_latents` x `grid.k` with `grid.seed`, the main configuration
(`main`, fixed in advance) with `main.seeds`, and `controls` (m = 1024, K = 4: L0
matched to the k = 16 SAE) and `undercomplete` (m = 16, K = 4: the k = 16 SAE's
dictionary size, to separate it from the L1 vs. TopK objective). The k = 16 SAE is only read (data check
and its holdout row of `training_grid.csv`).

Outputs in `topk_dir`: `models/topk_m<m>_k<K>_seed<s>.pt`, `training_grid.csv` (holdout
metrics), `training_history.csv`, `latent_frequency_holdout.csv`, `data_check.json`,
`figures/training_curves.png`.
"""

import hashlib
import json
from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch import Tensor

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import Activations, load_activations
from nuclei_graph.xai.metrics import LinearHead
from nuclei_graph.xai.runtime import setup_precision
from nuclei_graph.xai.sae import (
    ConceptAutoencoder,
    covariance,
    encode,
    load_autoencoder,
)
from nuclei_graph.xai.topk import (
    TopKAutoencoder,
    load_topk,
    save_topk,
    sparse_decode,
    train_topk_sae,
)
from xai.concept_masks import code_commit
from xai.train_sae import holdout_split, load_part


def sha256(path: Path, block: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def data_check(
    config: DictConfig,
    slides: pd.DataFrame,
    validation: Activations,
    test: Activations,
    holdout_ids: pd.Series,
    sampler_stats: dict[str, dict[str, float]],
    reference_holdout_re: float,
) -> dict:
    """Split, slide lists and h files against those of the k = 16 SAE; asserts the hard checks."""
    sae_dir = Path(config.sae_dir)
    stored = pd.read_csv(sae_dir / "holdout_slides.csv")["slide_id"]
    split_ids = {s: set(g["slide_id"]) for s, g in slides.groupby("split")}
    check: dict = {
        "slides_per_split": {s: len(ids) for s, ids in split_ids.items()},
        "holdout_slides": len(holdout_ids),
        "holdout_equals_k16_holdout_slides_csv": stored.tolist()
        == holdout_ids.tolist(),
        "holdout_within_validation": set(holdout_ids) <= split_ids["validation"],
        "validation_index_slides_equal_split": set(validation.index["slide_id"])
        == split_ids["validation"],
        "test_index_slides_equal_split": set(test.index["slide_id"])
        == split_ids["test"],
        "validation_nuclei": len(validation),
        "test_nuclei": len(test),
    }

    # the k = 16 training run logged its fit/holdout sampler statistics
    logged = MlflowClient().get_run(config.reference.run_id).data.metrics
    deviations = {
        f"sampler/{part}/{key}": abs(value - logged[f"sampler/{part}/{key}"])
        for part, stats in sampler_stats.items()
        for key, value in stats.items()
    }
    check["sampler_stats_max_abs_deviation_from_k16_run"] = max(deviations.values())
    check["sampler_stats_compared"] = len(deviations)

    files = {}
    model_mtime = Path(config.reference.model).stat().st_mtime
    for split, activations in (("validation", validation), ("test", test)):
        for name in ("h", "logits"):
            path = Path(config.activations_dir) / split / f"{name}.npy"
            array = activations.h if name == "h" else activations.logits
            files[f"{split}/{name}.npy"] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "modified_before_k16_sae": path.stat().st_mtime < model_mtime,
            }
    check["files"] = files

    # the k = 16 SAE on these files reproduces its stored holdout RE and test codes
    models = pd.read_csv(sae_dir / "models.csv")
    stored_re = float(
        models.loc[models["path"] == Path(config.reference.model).name, "RE"].iloc[0]
    )
    check["k16_holdout_RE_stored"] = stored_re
    check["k16_holdout_RE_recomputed"] = reference_holdout_re
    reference, _ = load_autoencoder(config.reference.model)
    rows = np.sort(
        np.random.default_rng(config.seed).choice(
            len(test), config.reference.check_rows, replace=False
        )
    )
    codes = encode(reference, np.asarray(test.h[rows])).numpy()
    stored_codes = np.load(config.reference.codes_test, mmap_mode="r")
    check["k16_codes_test_rows"] = int(stored_codes.shape[0])
    check["k16_codes_test_max_abs_deviation"] = float(
        np.abs(codes - stored_codes[rows]).max()
    )
    check["k16_codes_test_check_rows"] = len(rows)

    assert check["holdout_slides"] == 40, check
    for key in (
        "holdout_equals_k16_holdout_slides_csv",
        "holdout_within_validation",
        "validation_index_slides_equal_split",
        "test_index_slides_equal_split",
    ):
        assert check[key], key
    assert sampler_stats["fit"]["nuclei"] == config.reference.fit_nuclei
    assert sampler_stats["holdout"]["nuclei"] == config.reference.holdout_nuclei
    assert check["sampler_stats_max_abs_deviation_from_k16_run"] < 1e-9, deviations
    assert check["test_nuclei"] == stored_codes.shape[0] == 14_730_504
    assert all(f["modified_before_k16_sae"] for f in files.values())
    assert abs(reference_holdout_re - stored_re) < 1e-4, (
        reference_holdout_re,
        stored_re,
    )
    assert check["k16_codes_test_max_abs_deviation"] < 1e-2  # TF32 (stored) vs CPU fp32
    return check


@torch.no_grad()
def holdout_metrics(
    model: ConceptAutoencoder,
    z: Tensor,
    weights: np.ndarray,
    head: LinearHead,
    f: np.ndarray,
    total_variance: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Holdout metrics as `xai.train_sae.holdout_metrics`, plus FE and frequencies.

    Unweighted (and L0, RE weighted); also the explained variance, the normalized MSE
    ‖h - ĥ‖² / ‖h‖² of the training objective and the per-latent frequencies.
    """
    sq_err, sq_norm, f_a, l0 = [], [], [], []
    counts = np.zeros(model.num_concepts, dtype=np.int64)
    for i in range(0, len(z), 65536):
        zb = z[i : i + 65536].float()
        if isinstance(model, TopKAutoencoder):
            top, values = model.topk(zb)
            z_hat = (
                sparse_decode(top, values, model.decoder_weight) + model.decoder_bias
            )
            active = values > 0
            counts += np.bincount(top[active].numpy(), minlength=model.num_concepts)
            l0.append(active.sum(1).numpy())
        else:
            codes = encode(model, zb)
            z_hat = model.decode(codes)
            counts += (codes != 0).sum(0).numpy()
            l0.append((codes != 0).sum(1).numpy())
        z64, z_hat64 = zb.double().numpy(), z_hat.double().numpy()
        sq_err.append(((z64 - z_hat64) ** 2).sum(1))
        sq_norm.append((z64**2).sum(1))
        f_a.append(head(z_hat64))
    sq_err, l0, f_a = np.concatenate(sq_err), np.concatenate(l0), np.concatenate(f_a)
    frequency = counts / len(z)
    re = float(np.sqrt(sq_err.mean()))
    return {
        "L0": float(l0.mean()),
        "dead_fraction": float((counts == 0).mean()),
        "RE": re,
        "explained_variance": 1 - re**2 / total_variance,
        "normalized_mse": float(sq_err.sum() / np.concatenate(sq_norm).sum()),
        "FE": float(np.sqrt(((f - f_a) ** 2).mean())),
        "L0_weighted": float(l0 @ weights / weights.sum()),
        "RE_weighted": float(np.sqrt(sq_err @ weights / weights.sum())),
        "frequency_median": float(np.median(frequency)),
        "frequency_max": float(frequency.max()),
        "share_latents_frequency_above_10pct": float((frequency > 0.1).mean()),
        "share_latents_frequency_below_1e-4": float((frequency < 1e-4).mean()),
    }, frequency


def runs(config: DictConfig) -> list[tuple[int, int, int]]:
    """(m, K, seed): main seeds, the rest of the grid, controls, undercomplete models."""
    main = [(config.main.num_latents, config.main.k, s) for s in config.main.seeds]
    grid = [
        (m, k, config.grid.seed) for m in config.grid.num_latents for k in config.grid.k
    ]
    controls = [
        (c.num_latents, c.k, c.seed) for c in [*config.controls, *config.undercomplete]
    ]
    return main + [r for r in sorted(grid) + controls if r not in main]


def plot_curves(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, group in history.groupby("model", sort=False):
        line = axes[0].plot(group["epoch"], group["val_mse"], label=name)[0]
        axes[0].plot(group["epoch"], group["train_mse"], ls=":", color=line.get_color())
        axes[1].plot(group["epoch"], group["train_aux"], color=line.get_color())
        axes[2].plot(group["epoch"], group["dead_fraction_end"], color=line.get_color())
    axes[0].set(
        title="MSE ‖h_n - ĥ_n‖² (solid: holdout, weighted; dotted: train)",
        xlabel="epoch (2M samples)",
        yscale="log",
    )
    axes[1].set(title="AuxK loss (train)", xlabel="epoch", yscale="symlog")
    axes[2].set(title="dead fraction (no firing in 1M samples)", xlabel="epoch")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@with_cli_args(["+xai=train_topk_sae"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    for key, value in setup_precision(config.matmul_precision).items():
        client.set_tag(logger.run_id, key, value)
    torch.set_num_threads(config.num_threads)
    commit, dirty = code_commit("xai", "nuclei_graph", "configs")
    client.set_tag(logger.run_id, "xai.reference_sae_run_id", config.reference.run_id)
    client.set_tag(
        logger.run_id, "xai.reference_sae_model", Path(config.reference.model).name
    )
    client.set_tag(
        logger.run_id, "xai.code_commit", f"{commit}{' (dirty)' if dirty else ''}"
    )
    client.set_tag(logger.run_id, "xai.num_threads", str(config.num_threads))
    for source in (
        __file__,
        "nuclei_graph/xai/topk.py",
        "configs/xai/train_topk_sae.yaml",
    ):
        logger.log_artifact(
            str(Path(__file__).parents[1] / source), "code"
        )  # may be uncommitted
    torch.use_deterministic_algorithms(True, warn_only=True)
    topk_dir = Path(config.topk_dir)
    (topk_dir / "models").mkdir(parents=True, exist_ok=True)
    (topk_dir / "figures").mkdir(exist_ok=True)

    slides = pd.read_parquet(config.slides)
    holdout_ids = holdout_split(slides, config.holdout_size, config.seed)
    validation = load_activations(config.activations_dir, "validation")
    test = load_activations(config.activations_dir, "test")
    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    is_holdout = (
        validation.index["slide_id"].astype(str).isin(set(holdout_ids)).to_numpy()
    )
    z_fit, w_fit, stats_fit = load_part(
        validation, np.flatnonzero(~is_holdout), config.sampler.max_slide_share, "cpu"
    )
    z_holdout, w_holdout, stats_holdout = load_part(
        validation, np.flatnonzero(is_holdout), config.sampler.max_slide_share, "cpu"
    )
    holdout_rows = np.flatnonzero(is_holdout)
    f_holdout = head(np.asarray(validation.h[holdout_rows]))
    total_variance = float(np.trace(covariance(z_holdout)[1]))
    sampler_stats = {
        "fit": stats_fit | {"nuclei": len(z_fit)},
        "holdout": stats_holdout | {"nuclei": len(z_holdout)},
    }
    print(f"fit: {len(z_fit):,} nuclei, holdout: {len(z_holdout):,} nuclei")

    rows, frequencies = [], {}
    reference, reference_meta = load_autoencoder(config.reference.model)
    metrics, frequency = holdout_metrics(
        reference, z_holdout, w_holdout, head, f_holdout, total_variance
    )
    rows.append(
        {
            "model": "k16_l1_reference",
            "method": "sae_l1",
            "num_latents": 16,
            "k": np.nan,
            "seed": reference_meta["seed"],
            "best_epoch": reference_meta["best_epoch"],
            "epochs": reference_meta["epochs"],
            **metrics,
            "path": str(config.reference.model),
        }
    )
    frequencies["k16_l1_reference"] = frequency

    check = data_check(
        config, slides, validation, test, holdout_ids, sampler_stats, metrics["RE"]
    )
    check |= {"sampler": sampler_stats, "holdout_total_variance": total_variance}
    (topk_dir / "data_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps({k: v for k, v in check.items() if k != "files"}, indent=2))
    logger.log_metrics({"data_check/passed": 1.0})
    logger.log_artifact(str(topk_dir / "data_check.json"), "training")

    training = OmegaConf.to_container(config.training, resolve=True)
    histories = []
    for num_latents, k, seed in runs(config):
        name = f"topk_m{num_latents}_k{k}_seed{seed}"
        path = topk_dir / "models" / f"{name}.pt"
        if config.reuse_existing and path.exists():
            model, meta = load_topk(str(path))
            assert (meta["num_latents"], meta["k"], meta["seed"]) == (
                num_latents,
                k,
                seed,
            )
            assert {**meta["training"], "device": None} == {
                **training,
                "device": None,
            }, f"{path}: other training settings"
        else:

            def on_epoch(entry: dict[str, float], name: str = name) -> None:
                logger.log_metrics(
                    {
                        f"{name}/{key}": value
                        for key, value in entry.items()
                        if key != "epoch"
                    },
                    step=entry["epoch"],
                )
                print(
                    name,
                    {key: round(value, 5) for key, value in entry.items()},
                    flush=True,
                )

            result = train_topk_sae(
                z_fit,
                w_fit,
                z_holdout,
                w_holdout,
                num_latents=num_latents,
                k=k,
                seed=seed,
                on_epoch=on_epoch,
                **training,
            )
            model = result.model
            meta = {
                "num_latents": num_latents,
                "k": k,
                "seed": seed,
                "scale": float(model.scale),
                "best_epoch": result.best_epoch,
                "epochs": len(result.history),
                "holdout_mse": result.best_val_loss,
                "training": training,
                "history": result.history,
            }
            save_topk(model, str(path), **meta)
        histories.append(pd.DataFrame(meta["history"]).assign(model=name))
        metrics, frequency = holdout_metrics(
            model, z_holdout, w_holdout, head, f_holdout, total_variance
        )
        frequencies[name] = frequency
        rows.append(
            {
                "model": name,
                "method": "topk",
                "num_latents": num_latents,
                "k": k,
                "seed": seed,
                "main": (num_latents, k) == (config.main.num_latents, config.main.k),
                "scale": meta["scale"],
                "best_epoch": meta["best_epoch"],
                "epochs": meta["epochs"],
                "stopped_early": meta["epochs"] < config.training.max_epochs,
                "holdout_mse_weighted_normalized": meta["holdout_mse"],
                "final_train_dead_fraction": meta["history"][-1]["dead_fraction_end"],
                **metrics,
                "path": str(path.relative_to(topk_dir)),
            }
        )
        logger.log_metrics(
            {f"holdout/{name}/{key}": float(value) for key, value in metrics.items()}
        )
        print(
            f"{name}: RE={metrics['RE']:.3f} EV={metrics['explained_variance']:.3f} "
            f"FE={metrics['FE']:.3f} L0={metrics['L0']:.2f} dead={metrics['dead_fraction']:.3f} "
            f"best epoch {meta['best_epoch']} of {meta['epochs']}"
        )
        pd.DataFrame(rows).to_csv(topk_dir / "training_grid.csv", index=False)

    history = pd.concat(histories, ignore_index=True)
    history.to_csv(topk_dir / "training_history.csv", index=False)
    pd.DataFrame(
        [
            {"model": name, "latent": j, "frequency": value}
            for name, freq in frequencies.items()
            for j, value in enumerate(freq)
        ]
    ).to_csv(topk_dir / "latent_frequency_holdout.csv", index=False)
    plot_curves(history, topk_dir / "figures" / "training_curves.png")
    for name in (
        "training_grid.csv",
        "training_history.csv",
        "latent_frequency_holdout.csv",
        "figures/training_curves.png",
    ):
        logger.log_artifact(str(topk_dir / name), "training")
    logger.log_artifacts(str(topk_dir / "models"), "models")


if __name__ == "__main__":
    main()
