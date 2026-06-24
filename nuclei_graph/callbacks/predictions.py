import os
import tempfile

import mlflow
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from mlflow.tracking import MlflowClient

from nuclei_graph.nuclei_graph_typing import Batch, Outputs


class BasePredictionsCallback(Callback):
    def __init__(
        self,
        mlflow_artifact_path: str = "predictions",
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__()
        self.mlflow_artifact_path = mlflow_artifact_path
        self.mlflow_run_id = mlflow_run_id
        self.tmp_dir: tempfile.TemporaryDirectory[str] | None = None

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

    def _save_parquet(self, df: pd.DataFrame, slide_id: str) -> None:
        if self.tmp_dir is not None:
            output_path = os.path.join(self.tmp_dir.name, f"{slide_id}.parquet")
            df.to_parquet(output_path, index=False, engine="pyarrow")

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.tmp_dir is not None:
            mlflow_run_id = self.mlflow_run_id

            if mlflow_run_id is None:
                active_run = mlflow.active_run()
                if active_run is not None:
                    mlflow_run_id = active_run.info.run_id

            assert mlflow_run_id is not None

            MlflowClient().log_artifacts(
                run_id=mlflow_run_id,
                local_dir=self.tmp_dir.name,
                artifact_path=self.mlflow_artifact_path,
            )

            self.tmp_dir.cleanup()
            self.tmp_dir = None


class NucleiPredictionCallback(BasePredictionsCallback):
    """Computes nucleus-level predictions.

    It saves a parquet file with nuclei IDs and prediction scores.
    """

    def __init__(
        self,
        mlflow_artifact_path: str = "predictions",
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(
            mlflow_artifact_path=mlflow_artifact_path, mlflow_run_id=mlflow_run_id
        )
        self.predictions: list[dict] = []

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        nuclei_logits = outputs["nuclei"].squeeze(-1)
        nuclei_preds = torch.sigmoid(nuclei_logits).cpu().numpy().flatten()

        metadata = batch["metadata"]
        assert metadata is not None, "Metadata is required to save predictions."

        slide_id = metadata["slide_id"][0]  # batch size is 1
        nuclei_ids = metadata["nuclei_ids"][0]  # batch size is 1

        preds_df = (
            pd.DataFrame({"id": nuclei_ids, "nuclei_prediction": nuclei_preds})
            .sort_values("id")
            .reset_index(drop=True)
        )
        self._save_parquet(preds_df, slide_id)


class CropPredictionCallback(BasePredictionsCallback):
    """Computes crop-level predictions.

    It saves a parquet file with nuclei IDs, nuclei and graph label predictions, and nuclei attention scores.
    """

    def __init__(
        self,
        mlflow_artifact_path: str = "predictions",
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(
            mlflow_artifact_path=mlflow_artifact_path, mlflow_run_id=mlflow_run_id
        )
        self.predictions: list[dict] = []

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch["metadata"]
        assert metadata is not None, "Metadata is required to save predictions."

        slide_id = metadata["slide_id"][0]
        nuclei_ids = metadata["nuclei_ids"][0]

        nuclei_preds = torch.sigmoid(outputs["nuclei"].squeeze(-1))
        attn_scores = outputs["attn_weights"].squeeze(-1)
        graph_pred = torch.sigmoid(outputs["graph"].view(-1)[0])

        df = pd.DataFrame(
            {
                "id": nuclei_ids,  # batch size is 1
                "nuclei_prediction": nuclei_preds.cpu().numpy().flatten(),
                "attention_score": attn_scores.cpu().numpy().flatten(),
                "graph_prediction": graph_pred.item(),
            }
        )
        df = df.sort_values("id").reset_index(drop=True)

        self._save_parquet(df, slide_id)


class TilePredictionCallback(BasePredictionsCallback):
    """Computes tile-level predictions.

    Accumulates predictions across batches and saves a parquet file per slide
    containing x, y coordinates and the graph prediction score.
    """

    def __init__(
        self,
        mlflow_artifact_path: str = "predictions",
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(
            mlflow_artifact_path=mlflow_artifact_path, mlflow_run_id=mlflow_run_id
        )
        self.predictions: list[dict] = []

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits_graph = outputs["graph"].view(-1)
        preds_graph = torch.sigmoid(logits_graph).cpu().numpy()

        metadata = batch["metadata"]
        assert metadata is not None, "Metadata is required"
        slide_ids = metadata["slide_id"]

        for i in range(len(slide_ids)):
            self.predictions.append(
                {
                    "slide_id": slide_ids[i],
                    "x": metadata["x"][i],
                    "y": metadata["y"][i],
                    "tile_prediction": preds_graph[i],
                }
            )

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.predictions:
            df = pd.DataFrame(self.predictions)

            for slide_id, group_df in df.groupby("slide_id"):
                clean_df = group_df.drop(columns=["slide_id"]).reset_index(drop=True)
                self._save_parquet(clean_df, str(slide_id))

            self.predictions.clear()

        super().on_predict_epoch_end(trainer, pl_module)


class NucleiToTilePredictionCallback(BasePredictionsCallback):
    """Computes tile-level predictions by aggregating nuclei-level predictions.

    Supports 'max', 'mean', and 'top_k' pooling strategies.
    """

    def __init__(
        self,
        pooling_mode: str = "top_k",
        k: int = 10,
        mlflow_artifact_path: str = "predictions",
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(
            mlflow_artifact_path=mlflow_artifact_path, mlflow_run_id=mlflow_run_id
        )
        assert pooling_mode in ["max", "mean", "top_k"], "Invalid pooling mode."
        self.pooling_mode = pooling_mode
        self.k = k
        self.predictions: list[dict] = []

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        nuclei_logits = outputs["nuclei"].squeeze(-1)
        nuclei_preds = torch.sigmoid(nuclei_logits)

        metadata = batch["metadata"]
        assert metadata is not None, "Metadata is required"
        slide_ids = metadata["slide_id"]

        seq_lens_tensor = batch["seq_lens"]
        seq_lens_list = seq_lens_tensor.tolist()

        preds_split = torch.split(nuclei_preds, seq_lens_list)

        for i, valid_preds in enumerate(preds_split):
            if len(valid_preds) == 0:
                tile_pred = 0.0
            else:
                if self.pooling_mode == "max":
                    tile_pred = valid_preds.max().item()

                elif self.pooling_mode == "mean":
                    tile_pred = valid_preds.mean().item()

                elif self.pooling_mode == "top_k":
                    actual_k = min(self.k, len(valid_preds))
                    top_k_preds, _ = torch.topk(valid_preds, actual_k)
                    tile_pred = top_k_preds.mean().item()

            self.predictions.append(
                {
                    "slide_id": slide_ids[i],
                    "x": metadata["x"][i],
                    "y": metadata["y"][i],
                    "tile_prediction": float(tile_pred),
                }
            )

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.predictions:
            df = pd.DataFrame(self.predictions)

            for slide_id, group_df in df.groupby("slide_id"):
                clean_df = group_df.drop(columns=["slide_id"]).reset_index(drop=True)
                self._save_parquet(clean_df, str(slide_id))

            self.predictions.clear()

        super().on_predict_epoch_end(trainer, pl_module)
