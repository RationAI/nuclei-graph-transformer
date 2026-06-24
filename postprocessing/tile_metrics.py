from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import ConfusionMatrixDisplay
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


def get_predictions(predictions_dir: Path) -> pd.DataFrame:
    all_preds = []
    for parquet_path in predictions_dir.rglob("*.parquet"):
        slide_id = parquet_path.stem
        slide_pred_df = pd.read_parquet(parquet_path)
        slide_pred_df["slide_id"] = slide_id
        all_preds.append(slide_pred_df)
    return pd.concat(all_preds, ignore_index=True)


@with_cli_args(["+postprocessing=metrics"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    predictions_dir = Path(download_artifacts(config.predictions_uri))

    tiles_df = pd.read_parquet(download_artifacts(config.metadata_uri))
    slides_df = pd.read_parquet(download_artifacts(config.slides_uri))

    id_to_stem = dict(zip(slides_df["id"], slides_df["stem"], strict=True))
    tiles_df["slide_id"] = tiles_df["slide_id"].map(id_to_stem)

    carcinoma_roi_t = config.carcinoma_roi_t
    tiles_df["carcinoma"] = (
        tiles_df["carcinoma_roi_percentage"] > carcinoma_roi_t
    ).astype(int)

    preds_df = get_predictions(predictions_dir)
    merged_df = pd.merge(preds_df, tiles_df, on=["slide_id", "x", "y"], how="inner")

    target_col = config.label_column
    merged_df[target_col] = merged_df[target_col].fillna(0).astype(int)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        slide_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(config.threshold),
                "precision": BinaryPrecision(config.threshold),
                "recall": BinaryRecall(config.threshold),
                "specificity": BinarySpecificity(config.threshold),
                "npv": BinaryNegativePredictiveValue(config.threshold),
            }
        )

        slide_results = []
        for slide_id, group in merged_df.groupby("slide_id"):
            preds_t = torch.tensor(group["tile_prediction"].values)
            targets_t = torch.tensor(group[target_col].values).long()

            computed = slide_metrics(preds_t, targets_t)
            res = {"slide_id": slide_id, **{k: float(v) for k, v in computed.items()}}
            slide_results.append(res)
            slide_metrics.reset()

        slide_results_df = pd.DataFrame(slide_results)

        logger.log_table(
            data={
                str(k): v for k, v in slide_results_df.to_dict(orient="list").items()
            },
            artifact_file=config.mlflow_artifact_path,
        )

        global_tile_metrics = MetricCollection(
            {
                "AUPRC": BinaryAveragePrecision(),
                "AUROC": BinaryAUROC(),
                "precision": BinaryPrecision(config.threshold),
                "recall": BinaryRecall(config.threshold),
                "accuracy": BinaryAccuracy(config.threshold),
                "specificity": BinarySpecificity(config.threshold),
                "npv": BinaryNegativePredictiveValue(config.threshold),
                "confusion_matrix": BinaryConfusionMatrix(config.threshold),
            },
            prefix="test_thresholded/",
        )

        preds_t = torch.tensor(merged_df["tile_prediction"].values)
        targets_t = torch.tensor(merged_df[target_col].values).long()
        computed_global_tiles = global_tile_metrics(preds_t, targets_t)

        numerical_metrics = {}
        for key, value in computed_global_tiles.items():
            if "confusion_matrix" in key:
                disp_cf = ConfusionMatrixDisplay(
                    confusion_matrix=value.cpu().numpy(),
                    display_labels=["Negative", "Positive"],
                )
                fig, ax = plt.subplots(figsize=(6, 5))
                disp_cf.plot(ax=ax, cmap="Blues", colorbar=False)
                plt.title("Global Tile-Level Confusion Matrix")
                plt.tight_layout()

                fig_path = tmp_path / "tile_confusion_matrix.png"
                fig.savefig(fig_path, dpi=600)
                logger.log_artifact(local_path=str(fig_path), artifact_path="plots")
                plt.close(fig)
            else:
                numerical_metrics[key] = float(value)

        if numerical_metrics:
            logger.log_metrics(numerical_metrics)


if __name__ == "__main__":
    main()
