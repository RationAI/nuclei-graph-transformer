import tempfile

import mlflow
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer

from nuclei_graph.nuclei_graph_typing import Outputs, PredictBatch


class WSLPredictionsCallback(Callback):
    """Computes nucleus-level predictions for the weakly supervised learning architecture.

    It saves a parquet file with nuclei IDs and prediction scores.
    """

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
        logits_permuted = outputs["nuclei"][0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        metadata = batch["metadata"][0]  # batch size is 1
        logits = logits_permuted[:seq_len][metadata["perm_inverse"]]

        preds_t = torch.sigmoid(logits).cpu().numpy().flatten()
        preds_df = pd.DataFrame({"id": metadata["nuclei_ids"], "prediction": preds_t})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/{metadata['slide_id']}.parquet"
            preds_df.to_parquet(output_path, index=False)

            active_run = mlflow.active_run()
            assert active_run is not None
            mlflow.log_artifacts(
                tmp_dir,
                artifact_path=self.mlflow_artifact_path,
                run_id=active_run.info.run_id,
            )


class MILPredictionsCallback(Callback):
    """Computes nucleus-level and graph-level predictions for the multiple instance learning architecture.

    It saves a parquet file with nuclei IDs, nuclei and graph label predictions, and nuclei attention scores.
    Additionally, a CSV file is saved with misclassified slides based on the graph-level predictions.
    """

    def __init__(
        self, threshold: float, mlflow_artifact_path: str = "predictions"
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.mlflow_artifact_path = mlflow_artifact_path
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
        logits_permuted = outputs["nuclei"][0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        metadata = batch["metadata"][0]  # batch size is 1
        logits = logits_permuted[:seq_len][metadata["perm_inverse"]]

        attn_permuted = outputs["attn_weights"][0].squeeze(-1)  # (n,)
        attn_scores = attn_permuted[:seq_len][metadata["perm_inverse"]]

        graph_pred = torch.sigmoid(outputs["graph"][0]).item()
        df = pd.DataFrame(
            {
                "id": metadata["nuclei_ids"],
                "nuclei_prediction": torch.sigmoid(logits).cpu().numpy().flatten(),
                "attention_score": attn_scores.cpu().numpy().flatten(),
                "graph_prediction": graph_pred,
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

        targets_graph = batch["slides"]["y"]["graph"]

        # track the slide-level predictions for the misclassification logs
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = f"{tmp_dir}/misclassifications.csv"
            misclassif_df.to_csv(csv_path, index=False)

            active_run = mlflow.active_run()
            assert active_run is not None
            mlflow.log_artifacts(
                tmp_dir,
                artifact_path=self.mlflow_artifact_path,
                run_id=active_run.info.run_id,
            )

        self.slide_preds = {"slide_id": [], "is_carcinoma": [], "prediction": []}
