"""Computes nuclei- and slide-level classification metrics from crop predictions.

Loads per-slide crop prediction parquets, merges them with nuclei-level
supervision and slide metadata, and logs:

- Per-slide nuclei-level metrics (accuracy, precision, recall, specificity,
  negative predictive value), logged as a table.
- Global nuclei-level metrics (the above plus AUPRC/AUROC), logged as scalar
  metrics along with 95% confidence intervals computed via slide-level bootstrapping.
- Slide-level graph metrics derived from a single `graph_prediction` per
  slide, including a confusion matrix plot and a CSV of misclassified slides;
  skipped if no `graph_prediction` column is present.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    roc_auc_score,
)
from sklearn.utils import resample
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryNegativePredictiveValue,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from tqdm import tqdm

from nuclei_graph.mlflow_utils import tag_parent_run
from postprocessing.mlflow_utils import setup_mlflow


def get_predictions(slide_ids: pd.Series, predictions_dir: Path) -> pd.DataFrame:
    """Loads all parquet predictions and concatenates them."""
    all_preds = []
    for slide_id in slide_ids.unique():
        parquet_path = predictions_dir / f"{slide_id}.parquet"
        if not parquet_path.exists():
            continue
        slide_pred_df = pd.read_parquet(parquet_path)
        slide_pred_df["slide_id"] = slide_id
        all_preds.append(slide_pred_df)

    return pd.concat(all_preds, ignore_index=True)


def load_and_merge_data(config: DictConfig, predictions_dir: Path) -> pd.DataFrame:
    """Downloads artifacts, loads parquets, and merges predictions with ground truth."""
    metadata_df = pd.read_parquet(download_artifacts(config.metadata_uri))
    supervision_df = pd.read_parquet(config.supervision_dir)

    preds_df = get_predictions(metadata_df["slide_id"], predictions_dir)

    merged_df = pd.merge(preds_df, supervision_df, on=["slide_id", "id"], how="left")
    merged_df = pd.merge(
        merged_df,
        metadata_df[["slide_id", "is_carcinoma", "slide_path"]],
        on="slide_id",
        how="left",
    )

    label_col = config.label_column
    merged_df[label_col] = merged_df[label_col].fillna(0).astype(int)
    merged_df.loc[~merged_df["is_carcinoma"], label_col] = 0

    return merged_df


def log_per_slide_nuclei_metrics(
    merged_df: pd.DataFrame,
    config: DictConfig,
    client: MlflowClient,
    run_id: str | None,
) -> None:
    """Calculates and logs nuclei-level metrics grouped by individual slides."""
    slide_metrics = MetricCollection(
        {
            "accuracy": BinaryAccuracy(config.threshold),
            "precision": BinaryPrecision(config.threshold),
            "recall": BinaryRecall(config.threshold),
            "specificity": BinarySpecificity(config.threshold),
            "negative_predictive_value": BinaryNegativePredictiveValue(
                config.threshold
            ),
        }
    )

    slide_results = []
    for slide_id, group in merged_df.groupby("slide_id"):
        preds_t = torch.tensor(group["nuclei_prediction"].values)
        targets_t = torch.tensor(group[config.label_column].values).long()

        computed = slide_metrics(preds_t, targets_t)
        slide_results.append(
            {"slide_id": slide_id, **{k: float(v) for k, v in computed.items()}}
        )
        slide_metrics.reset()

    if run_id is not None:
        slide_results_df = pd.DataFrame(slide_results)
        table_data = {
            str(k): v for k, v in slide_results_df.to_dict(orient="list").items()
        }
        client.log_table(
            run_id=run_id, data=table_data, artifact_file=config.mlflow_artifact_path
        )


def compute_bootstrapped_cis(
    merged_df: pd.DataFrame, label_col: str, pred_col: str, n_iterations: int = 1000
) -> dict[str, float]:
    """Computes 95% CIs using fast slide-level bootstrapping."""
    slide_groups = [group for _, group in merged_df.groupby("slide_id")]

    auroc_scores = []
    auprc_scores = []

    for _ in tqdm(range(n_iterations), desc="Bootstrapping CIs"):
        resampled_groups = resample(slide_groups, n_samples=len(slide_groups))
        assert resampled_groups is not None

        y_true = np.concatenate([g[label_col].values for g in resampled_groups])
        y_pred = np.concatenate([g[pred_col].values for g in resampled_groups])

        # Ensure we have both classes in the resample to avoid metric crashes
        if len(np.unique(y_true)) < 2:
            continue

        auroc_scores.append(roc_auc_score(y_true, y_pred))
        auprc_scores.append(average_precision_score(y_true, y_pred))

    return {
        "AUROC_lower_95": np.percentile(auroc_scores, 2.5),
        "AUROC_upper_95": np.percentile(auroc_scores, 97.5),
        "AUPRC_lower_95": np.percentile(auprc_scores, 2.5),
        "AUPRC_upper_95": np.percentile(auprc_scores, 97.5),
    }


def log_global_nuclei_metrics(
    merged_df: pd.DataFrame,
    config: DictConfig,
    client: MlflowClient,
    run_id: str | None,
) -> None:
    """Calculates and logs global nuclei-level metrics and their bootstrapped CIs."""
    global_nuclei_metrics = MetricCollection(
        {
            "AUPRC": BinaryAveragePrecision(),
            "AUROC": BinaryAUROC(),
            "precision": BinaryPrecision(config.threshold),
            "recall": BinaryRecall(config.threshold),
            "accuracy": BinaryAccuracy(config.threshold),
            "specificity": BinarySpecificity(config.threshold),
        },
        prefix="test_thresholded/nuclei_",
    )

    preds_t = torch.tensor(merged_df["nuclei_prediction"].values)
    targets_t = torch.tensor(merged_df[config.label_column].values).long()
    computed = global_nuclei_metrics(preds_t, targets_t)

    # Calculate Bootstrapped Confidence Intervals
    cis = compute_bootstrapped_cis(
        merged_df=merged_df,
        label_col=config.label_column,
        pred_col="nuclei_prediction",
        n_iterations=2000,
    )

    if run_id is not None:
        # Log standard metrics
        for k, v in computed.items():
            client.log_metric(run_id, k, float(v))

        # Log confidence intervals
        for k, v in cis.items():
            client.log_metric(run_id, f"test_thresholded/nuclei_{k}", float(v))


def log_slide_level_graph_metrics(
    merged_df: pd.DataFrame,
    config: DictConfig,
    client: MlflowClient,
    run_id: str | None,
    tmp_path: Path,
) -> None:
    """Calculates slide-level predictions, logs misclassifications, and plots the confusion matrix."""
    if "graph_prediction" not in merged_df.columns:
        return

    graph_df = merged_df.drop_duplicates(subset=["slide_id"]).copy()
    graph_df["predicted_class"] = (
        graph_df["graph_prediction"] >= config.threshold
    ).astype(bool)

    misclassif_df = graph_df[graph_df["predicted_class"] != graph_df["is_carcinoma"]]
    csv_path = tmp_path / "graph_misclassifications.csv"
    misclassif_df[
        [
            "slide_id",
            "graph_prediction",
            "predicted_class",
            "is_carcinoma",
            "slide_path",
        ]
    ].to_csv(csv_path, index=False)

    if run_id is not None:
        client.log_artifact(run_id, str(csv_path))

    graph_metrics = MetricCollection(
        {
            "AUPRC": BinaryAveragePrecision(),
            "AUROC": BinaryAUROC(),
            "precision": BinaryPrecision(config.threshold),
            "recall": BinaryRecall(config.threshold),
            "accuracy": BinaryAccuracy(config.threshold),
            "specificity": BinarySpecificity(config.threshold),
            "confusion_matrix": BinaryConfusionMatrix(config.threshold),
        },
        prefix="test_thresholded/graph_",
    )

    preds_t = torch.tensor(graph_df["graph_prediction"].values)
    targets_t = torch.tensor(graph_df["is_carcinoma"].values).long()
    computed_graph = graph_metrics(preds_t, targets_t)

    numerical_metrics = {}
    for key, value in computed_graph.items():
        if "confusion_matrix" in key:
            disp_cf = ConfusionMatrixDisplay(
                confusion_matrix=value.cpu().numpy(),
                display_labels=["Negative", "Positive"],
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            disp_cf.plot(ax=ax, cmap="Blues", colorbar=False)
            plt.title("Slide-Level Confusion Matrix")
            plt.tight_layout()

            fig_path = tmp_path / "confusion_matrix.png"
            fig.savefig(fig_path, dpi=1200)

            if run_id is not None:
                client.log_artifact(run_id, str(fig_path))
            plt.close(fig)
        else:
            numerical_metrics[key] = float(value)

    if run_id is not None and numerical_metrics:
        for k, v in numerical_metrics.items():
            client.log_metric(run_id, k, float(v))


@with_cli_args(["+postprocessing=crop/metrics"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert logger.run_id is not None
    tag_parent_run(logger.run_id, config.get("mlflow_parent_run_id"))

    client, mlflow_run_id = setup_mlflow(config)

    predictions_dir = Path(download_artifacts(config.predictions_uri))
    merged_df = load_and_merge_data(config, predictions_dir)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        log_per_slide_nuclei_metrics(merged_df, config, client, mlflow_run_id)
        log_global_nuclei_metrics(merged_df, config, client, mlflow_run_id)
        log_slide_level_graph_metrics(
            merged_df, config, client, mlflow_run_id, tmp_path
        )


if __name__ == "__main__":
    main()
