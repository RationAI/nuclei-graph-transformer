import os
import tempfile

import mlflow
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer

from nuclei_graph.nuclei_graph_typing import Outputs, PredictBatch


class BasePredictionsCallback(Callback):
    def __init__(self, mlflow_artifact_path: str = "predictions") -> None:
        super().__init__()
        self.mlflow_artifact_path = mlflow_artifact_path
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
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.log_artifacts(
                    self.tmp_dir.name,
                    artifact_path=self.mlflow_artifact_path,
                    run_id=active_run.info.run_id,
                )
            self.tmp_dir.cleanup()
            self.tmp_dir = None


class WSLPredictionsCallback(BasePredictionsCallback):
    """Computes nucleus-level predictions.

    It saves a parquet file with nuclei IDs and prediction scores.
    """

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits = outputs["nuclei"].squeeze(-1)
        metadata = batch["metadata"][0]  # batch size is 1

        preds_t = torch.sigmoid(logits).cpu().numpy().flatten()
        preds_df = pd.DataFrame(
            {"id": metadata["nuclei_ids"], "nuclei_prediction": preds_t}
        )
        preds_df = preds_df.sort_values("id").reset_index(drop=True)

        self._save_parquet(preds_df, metadata["slide_id"])


class MILPredictionsCallback(BasePredictionsCallback):
    """Computes predictions for the MIL architecture.

    It saves a parquet file with nuclei IDs, nuclei and graph label predictions, and nuclei attention scores.
    """

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits = outputs["nuclei"].squeeze(-1)
        attn_scores = outputs["attn_weights"].squeeze(-1)

        graph_pred = torch.sigmoid(outputs["graph"][0]).item()

        metadata = batch["metadata"][0]  # batch size is 1
        nuclei_preds = torch.sigmoid(logits).cpu().numpy().flatten()

        df = pd.DataFrame(
            {
                "id": metadata["nuclei_ids"],
                "nuclei_prediction": nuclei_preds,
                "attention_score": attn_scores.cpu().numpy().flatten(),
                "graph_prediction": graph_pred,
            }
        )
        df = df.sort_values("id").reset_index(drop=True)

        self._save_parquet(df, metadata["slide_id"])
