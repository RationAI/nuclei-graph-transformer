import tempfile

import mlflow
import numpy as np
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from nuclei_graph.nuclei_graph_typing import PredictBatch
from scipy.interpolate import NearestNDInterpolator
from torch import Tensor


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
        nuclei_path = metadata["slide_nuclei_path"]
        nuclei = pd.read_parquet(nuclei_path, columns=["id", "centroid"])
        nuclei = nuclei.sort_values("id").reset_index(drop=True)

        logits = outputs[0].squeeze(-1)  # (n,)
        keep_indices = metadata["keep_indices"]
        logits_unpadded = logits[: len(keep_indices)]
        logits_ordered = logits_unpadded[metadata["perm_inverse"]]

        predicted_labels = torch.sigmoid(logits_ordered).cpu().numpy().flatten()
        nuclei.loc[keep_indices.cpu().numpy(), "prediction"] = predicted_labels

        # interpolate missing nuclei (eps-close neighbors dropped in the NucleiDataset)
        if nuclei["prediction"].isna().any():
            valid = nuclei.dropna(subset=["prediction"])
            coords = np.stack(valid["centroid"].tolist())
            interp = NearestNDInterpolator(coords, valid["prediction"].values)

            missing_mask = nuclei["prediction"].isna()
            missing_coords = np.stack(nuclei.loc[missing_mask, "centroid"].tolist())

            nuclei.loc[missing_mask, "prediction"] = interp(missing_coords)

        final_df = nuclei[["id", "prediction"]]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/{metadata['slide_id']}.parquet"
            final_df.to_parquet(output_path, index=False)

            active_run = mlflow.active_run()
            assert active_run is not None
            mlflow.log_artifacts(
                tmp_dir,
                artifact_path=self.mlflow_artifact_path,
                run_id=active_run.info.run_id,
            )
