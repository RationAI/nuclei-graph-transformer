import tempfile

import mlflow
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from nuclei_graph.nuclei_graph_typing import Outputs, PredictBatch


class WSLPredictionsCallback(Callback):
    def __init__(
        self,
        mlflow_artifact_path: str = "predictions",
    ) -> None:
        super().__init__()
        self.mlflow_artifact_path = mlflow_artifact_path

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch["metadata"][0]  # batch size is 1

        logits = outputs["nuclei"][0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        logits_unpadded = logits[:seq_len]
        logits_ordered = logits_unpadded[metadata["perm_inverse"]]

        predicted_labels = torch.sigmoid(logits_ordered).cpu().numpy().flatten()

        df = pd.DataFrame(
            {"id": metadata["nuclei_ids"], "prediction": predicted_labels}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/{metadata['slide_id']}.parquet"
            df.to_parquet(output_path, index=False)

            active_run = mlflow.active_run()
            assert active_run is not None
            mlflow.log_artifacts(
                tmp_dir,
                artifact_path=self.mlflow_artifact_path,
                run_id=active_run.info.run_id,
            )


class MILPredictionsCallback(Callback):
    def __init__(
        self,
        mlflow_artifact_path: str = "predictions",
    ) -> None:
        super().__init__()
        self.mlflow_artifact_path = mlflow_artifact_path

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch["metadata"][0]  # batch size is 1
        seq_len = batch["slides"]["seq_len"][0].item()

        logits = outputs["nuclei"][0].squeeze(-1)  # (n,)
        logits_unpadded = logits[:seq_len]
        logits_ordered = logits_unpadded[metadata["perm_inverse"]]
        nuclei_predicted_labels = torch.sigmoid(logits_ordered).cpu().numpy().flatten()

        attn = outputs["attn_weights"][0].squeeze(-1)  # (n,)
        attn_unpadded = attn[:seq_len]
        attn_ordered = attn_unpadded[metadata["perm_inverse"]]
        attn_scores = attn_ordered.cpu().numpy().flatten()

        df = pd.DataFrame(
            {
                "id": metadata["nuclei_ids"],
                "nuclei_prediction": nuclei_predicted_labels,
                "attention_score": attn_scores,
                "graph_prediction": torch.sigmoid(outputs["graph"][0]).item(),
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/{metadata['slide_id']}.parquet"
            df.to_parquet(output_path, index=False)

            active_run = mlflow.active_run()
            assert active_run is not None
            mlflow.log_artifacts(
                tmp_dir,
                artifact_path=self.mlflow_artifact_path,
                run_id=active_run.info.run_id,
            )
