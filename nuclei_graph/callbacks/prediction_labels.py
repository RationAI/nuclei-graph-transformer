import tempfile

import mlflow
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import PredictBatch


class PredictionsCallback(Callback):
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
        outputs: Tensor,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch["metadata"][0]  # batch size is 1

        logits = outputs[0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        logits_unpadded = logits[:seq_len]
        logits_ordered = logits_unpadded[metadata["perm_inverse"]]

        predicted_labels = torch.sigmoid(logits_ordered).cpu().numpy().flatten()

        df = pd.DataFrame(
            {"id": metadata["nuclei_ids"], "prediction": predicted_labels}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/{metadata['slide_id']}.parquet"
            df.to_parquet(output_path, index=False, columns=["id", "prediction"])

            active_run = mlflow.active_run()
            assert active_run is not None
            mlflow.log_artifacts(
                tmp_dir,
                artifact_path=self.mlflow_artifact_path,
                run_id=active_run.info.run_id,
            )
