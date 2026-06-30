"""Selects tile-level decision thresholds from pooled nuclei predictions on the validation split.

The pooling is swept over every mode in `config.pooling_modes` (`max`, `mean`, `top_k`); for
`top_k`, `k` is additionally swept over `config.k_values`.

For the best-performing combination, it derives three candidate thresholds:
- TPR threshold: lowest false-positive rate achieving a true-positive rate of 1.0.
- J threshold: threshold maximizing the Youden J statistic (TPR - FPR).
- F1 threshold: threshold maximizing the F1 score on the PR curve.

The curve plots and final thresholds are logged as metrics/artifacts; the full sweep is logged
as a single `pooling_mode_statistics.csv` table artifact.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from lightning import Callback, LightningModule
from lightning import Trainer as PLTrainer
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from nuclei_graph.data.datamodules.collator import GraphCollator
from nuclei_graph.data.datasets.crop.prediction import PredictionDataset
from nuclei_graph.data.datasets.tile.base import BaseTileDataset
from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.nuclei_graph_typing import Batch, Outputs
from postprocessing.mlflow_utils import setup_mlflow
from postprocessing.tile.pooling_utils import pool_slide_tiles


OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("add", lambda a, b: a + b)


def select_roc_thresholds(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> tuple[int, float, int, float]:
    """Picks the TPR=1.0 and Youden-J operating points off an ROC curve."""
    idx = np.where(np.isclose(tpr, 1.0))[0]
    if len(idx) > 0:
        tpr_idx = int(idx[np.argmin(fpr[idx])])
        tpr_threshold = float(thresholds[tpr_idx])
    else:
        tpr_idx = 0
        tpr_threshold = float("nan")

    j_idx = int(np.argmax(tpr - fpr))
    j_threshold = float(thresholds[j_idx])
    return tpr_idx, tpr_threshold, j_idx, j_threshold


def select_f1_threshold(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray
) -> tuple[int, float, float]:
    """Picks the F1-maximizing operating point off a precision-recall curve."""
    p, r = precision[:-1], recall[:-1]
    if len(p) == 0:
        return 0, float("nan"), float("nan")

    f1 = 2 * (p * r) / (p + r + 1e-8)
    best_idx = int(np.argmax(f1))
    return best_idx, float(thresholds[best_idx]), float(f1[best_idx])


def plot_roc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[plt.Figure, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    tpr_idx, tpr_threshold, j_idx, j_threshold = select_roc_thresholds(
        fpr, tpr, thresholds
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.scatter(
        fpr[tpr_idx],
        tpr[tpr_idx],
        color="red",
        label=f"TPR Thresh = {tpr_threshold:.3f}",
    )
    ax.scatter(
        fpr[j_idx], tpr[j_idx], color="green", label=f"J Thresh = {j_threshold:.3f}"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Tile-Level ROC")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()

    return fig, tpr_threshold, j_threshold


def plot_pr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[plt.Figure, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    best_idx, best_threshold, _ = select_f1_threshold(precision, recall, thresholds)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision)
    ax.scatter(
        recall[best_idx],
        precision[best_idx],
        color="red",
        label=f"F1 Thresh = {best_threshold:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Tile-Level PR Curve")
    ax.legend(loc="lower left")
    ax.grid(True)
    fig.tight_layout()

    return fig, best_threshold


class NucleiPredictionsCollector(Callback):
    """Collects per-nucleus predictions in memory instead of writing them to disk."""

    def __init__(self) -> None:
        super().__init__()
        self.predictions: dict[str, pd.Series] = {}

    def on_predict_batch_end(
        self,
        trainer: PLTrainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        nuclei_preds = (
            torch.sigmoid(outputs["nuclei"].squeeze(-1)).cpu().numpy().flatten()
        )

        metadata = batch["metadata"]
        assert metadata is not None
        stem = metadata["slide_id"][0]  # batch size is 1
        nuclei_ids = metadata["nuclei_ids"][0]

        self.predictions[stem] = (
            pd.DataFrame({"id": nuclei_ids, "nuclei_prediction": nuclei_preds})
            .sort_values("id")
            .set_index("id")["nuclei_prediction"]
        )


def predict_nuclei(
    config: DictConfig, dataset: BaseTileDataset, logger: MLFlowLogger
) -> dict[str, pd.Series]:
    """Runs the checkpointed nuclei model on the validation slides, in memory."""
    stems = dataset.tiles["stem"].unique()
    slides_df = dataset.metadata.loc[stems].reset_index()

    predict_dataset = PredictionDataset(
        metadata=slides_df,
        efd_order=config.data.dataset.efd_order,
        patch_size=config.get("patch_size"),
    )
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=GraphCollator(
            block_size=config.data.block_size, k=config.data.k, predict=True
        ),
    )

    model = instantiate(config.model)
    collector = NucleiPredictionsCollector()
    trainer = Trainer(logger=logger, callbacks=[collector])
    trainer.predict(model, dataloaders=predict_loader, ckpt_path=config.checkpoint)

    return collector.predictions


def pool_tiles(
    dataset: BaseTileDataset,
    nuclei_preds_by_stem: dict[str, pd.Series],
    pooling_mode: str,
    k: int,
) -> pd.DataFrame:
    """Pools nuclei predictions into tile scores over the validation split."""
    pooled = []
    for stem, tiles in dataset.tiles.groupby("stem"):
        tile_preds_df = pool_slide_tiles(
            tiles, dataset, nuclei_preds_by_stem[stem], pooling_mode, k
        )
        tile_preds_df["carcinoma"] = tiles["carcinoma"].to_numpy()
        pooled.append(tile_preds_df)

    return pd.concat(pooled, ignore_index=True)


def evaluate_pooling(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes cheap, plot-free summary statistics for a single pooling configuration."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    roc_auc = float(auc(fpr, tpr))
    _, tpr_threshold, _, j_threshold = select_roc_thresholds(fpr, tpr, roc_thresholds)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
    _, f1_threshold, f1_max = select_f1_threshold(precision, recall, pr_thresholds)

    return {
        "roc_auc": roc_auc,
        "tpr_threshold": tpr_threshold,
        "j_threshold": j_threshold,
        "f1_threshold": f1_threshold,
        "f1_max": f1_max,
    }


def sweep_pooling(
    config: DictConfig,
    dataset: BaseTileDataset,
    nuclei_preds_by_stem: dict[str, pd.Series],
) -> tuple[str, int | None, pd.DataFrame, pd.DataFrame]:
    """Pools tiles for every (pooling_mode, k) combination and picks the best by ROC AUC.

    `k` is only swept for `top_k`; `max`/`mean` ignore it and are evaluated once each.
    """
    merged_dfs: dict[tuple[str, int | None], pd.DataFrame] = {}
    rows = []

    for mode in config.pooling_modes:
        k_candidates: list[int | None] = (
            [int(k) for k in config.k_values] if mode == "top_k" else [None]
        )
        for k in k_candidates:
            merged_df = pool_tiles(dataset, nuclei_preds_by_stem, mode, k or 0)
            merged_dfs[mode, k] = merged_df

            metrics = evaluate_pooling(
                merged_df["carcinoma"].to_numpy(),
                merged_df["tile_prediction"].to_numpy(),
            )
            rows.append({"pooling_mode": mode, "k": k, **metrics})

    stats_df = pd.DataFrame(rows)
    best_idx = stats_df["roc_auc"].idxmax()
    best_mode = str(stats_df.loc[best_idx, "pooling_mode"])
    best_k_value = stats_df.loc[best_idx, "k"]
    best_k = int(best_k_value) if pd.notna(best_k_value) else None
    return best_mode, best_k, merged_dfs[best_mode, best_k], stats_df


@with_cli_args(["+postprocessing=tile/thresholds"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert logger.run_id is not None
    tag_parent_run(logger.run_id, config.get("mlflow_parent_run_id"))

    datamodule = instantiate(config.data, _recursive_=False)
    datamodule.setup("validate")
    dataset = datamodule.validation_dataset

    nuclei_preds_by_stem = predict_nuclei(config, dataset, logger)

    client, mlflow_run_id = setup_mlflow(config)
    assert mlflow_run_id is not None

    best_mode, best_k, merged_df, stats_df = sweep_pooling(
        config, dataset, nuclei_preds_by_stem
    )
    client.set_tag(mlflow_run_id, "thresholds/best_pooling_mode", best_mode)
    if best_k is not None:
        client.log_metric(mlflow_run_id, "thresholds/best_k", best_k)

    y_true = merged_df["carcinoma"].to_numpy()
    y_pred = merged_df["tile_prediction"].to_numpy()

    fig_roc, tpr_t, j_t = plot_roc(y_true, y_pred)
    fig_pr, f1_t = plot_pr(y_true, y_pred)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        roc_path = tmp_path / "tile_roc.png"
        pr_path = tmp_path / "tile_precision_recall.png"
        stats_path = tmp_path / "pooling_mode_statistics.csv"

        fig_roc.savefig(roc_path, dpi=1200)
        fig_pr.savefig(pr_path, dpi=1200)
        stats_df.to_csv(stats_path, index=False)

        client.log_artifact(mlflow_run_id, str(roc_path), "curves")
        client.log_artifact(mlflow_run_id, str(pr_path), "curves")
        client.log_artifact(mlflow_run_id, str(stats_path))

    client.log_metric(mlflow_run_id, "thresholds/tile_tpr", tpr_t)
    client.log_metric(mlflow_run_id, "thresholds/tile_j", j_t)
    client.log_metric(mlflow_run_id, "thresholds/tile_f1", f1_t)

    plt.close(fig_roc)
    plt.close(fig_pr)


if __name__ == "__main__":
    main()
