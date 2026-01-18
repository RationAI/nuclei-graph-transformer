import tempfile

import mlflow
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import PredictInput


class PredictionsCallback(Callback):
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: PredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, metadata = batch
        logits, metadata = outputs[0], metadata[0]  # batch size is 1 during inference

        perm_inverse = metadata["perm_inverse"]
        nuclei_ids = metadata["nuclei_ids"]

        logits_unpadded = logits[: len(nuclei_ids)]
        logits_original_order = logits_unpadded[perm_inverse]
        predicted_labels = torch.sigmoid(logits_original_order).cpu().numpy()
        df_predictions = pd.DataFrame({"id": nuclei_ids, "score": predicted_labels})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/{metadata['slide_id']}.parquet"
            df_predictions.to_parquet(output_path, index=False)

            active_run = mlflow.active_run()
            assert active_run is not None
            mlflow.log_artifacts(
                tmp_dir,
                artifact_path="predictions",
                run_id=active_run.info.run_id,
            )
