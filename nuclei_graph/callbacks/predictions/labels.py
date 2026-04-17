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
        self.tmp_dir = None

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

    def _save_parquet(self, df: pd.DataFrame, slide_id: str) -> None:
        if self.tmp_dir is not None:
            output_path = os.path.join(self.tmp_dir.name, f"{slide_id}.parquet")
            df.to_parquet(output_path, index=False)

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
        logits = outputs["nuclei"][0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        metadata = batch["metadata"][0]  # batch size is 1
        logits_ordered = logits[:seq_len][metadata["perm_inverse"]]

        preds_t = torch.sigmoid(logits_ordered).cpu().numpy().flatten()
        preds_df = pd.DataFrame({"id": metadata["nuclei_ids"], "prediction": preds_t})

        self._save_parquet(preds_df, metadata["slide_id"])


class MILPredictionsCallback(BasePredictionsCallback):
    """Computes nucleus-level and graph-level predictions for the MIL architecture.

    It saves a parquet file with nuclei IDs, nuclei and graph label predictions, and nuclei attention scores.
    Additionally, a CSV file is saved with misclassified slides based on the graph-level predictions.
    """

    def __init__(
        self, threshold: float, mlflow_artifact_path: str = "predictions"
    ) -> None:
        super().__init__(mlflow_artifact_path=mlflow_artifact_path)
        self.threshold = threshold
        self.slide_preds = {"slide_id": [], "is_carcinoma": [], "prediction": []}

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits = outputs["nuclei"][0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        metadata = batch["metadata"][0]  # batch size is 1
        logits_ordered = logits[:seq_len][metadata["perm_inverse"]]
        nuclei_preds = torch.sigmoid(logits_ordered).cpu().numpy().flatten()

        attn_permuted = outputs["attn_weights"][0].squeeze(-1)  # (n,)
        attn_scores = attn_permuted[:seq_len][metadata["perm_inverse"]]

        graph_pred = torch.sigmoid(outputs["graph"][0]).item()

        df = pd.DataFrame(
            {
                "id": metadata["nuclei_ids"],
                "nuclei_prediction": nuclei_preds,
                "attention_score": attn_scores.cpu().numpy().flatten(),
                "graph_prediction": graph_pred,
            }
        )
        self._save_parquet(df, metadata["slide_id"])

        targets_graph = batch["slides"]["y"]["graph"]

        assert targets_graph is not None
        self.slide_preds["slide_id"].append(metadata["slide_id"])
        self.slide_preds["is_carcinoma"].append(targets_graph.view(-1).item())
        self.slide_preds["prediction"].append(graph_pred)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        df = pd.DataFrame(self.slide_preds)
        df["predicted_class"] = (df["prediction"] >= self.threshold).astype(int)
        misclassif_df = df[df["predicted_class"] != df["is_carcinoma"]].copy()
        misclassif_df = misclassif_df.drop(columns=["predicted_class"])

        with tempfile.TemporaryDirectory() as csv_tmp_dir:
            csv_path = f"{csv_tmp_dir}/misclassifications.csv"
            misclassif_df.to_csv(csv_path, index=False)

            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.log_artifact(
                    local_path=csv_path,
                    run_id=active_run.info.run_id,
                )

        self.slide_preds = {"slide_id": [], "is_carcinoma": [], "prediction": []}

        super().on_predict_epoch_end(trainer, pl_module)
