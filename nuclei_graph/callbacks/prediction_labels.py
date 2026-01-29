import tempfile

import mlflow
import numpy as np
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from scipy.interpolate import NearestNDInterpolator
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import PredictBatch


class PredictionsCallback(Callback):
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

        slide_id = metadata["slide_id"]
        perm_inverse = metadata["perm_inverse"]
        nuclei_ids = metadata["nuclei_ids"]

        logits_unpadded = logits[: len(nuclei_ids)]
        logits_original_order = logits_unpadded[perm_inverse]
        predicted_labels = torch.sigmoid(logits_original_order).cpu().numpy().flatten()

        df_predictions = pd.DataFrame(
            {"id": nuclei_ids, "prediction": predicted_labels}
        )
        nuclei = pd.read_parquet(
            metadata["slide_nuclei_path"], columns=["id", "centroid"]
        )
        nuclei_preds = nuclei.merge(df_predictions, on="id", how="left")

        # some nuclei may not have predictions (nuclei that have a very close neighbor (< eps) are removed in NucleiDataset
        # due to assumptions in graph construction); we use nearest neighbor interpolation to fill in the missing predictions
        if nuclei_preds["prediction"].isna().any():
            valid = nuclei_preds.dropna(subset=["prediction"])
            coords = np.stack(valid["centroid"].tolist())
            vals = valid["prediction"].values
            interp = NearestNDInterpolator(coords, vals)
            missing_mask = nuclei_preds["prediction"].isna()
            missing_coords = np.stack(
                nuclei_preds.loc[missing_mask, "centroid"].tolist()
            )
            nuclei_preds.loc[missing_mask, "prediction"] = interp(missing_coords)

        final_df = nuclei_preds[["id", "prediction"]]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/{slide_id}.parquet"
            final_df.to_parquet(output_path, index=False)

            active_run = mlflow.active_run()
            assert active_run is not None
            mlflow.log_artifacts(
                tmp_dir,
                artifact_path="predictions",
                run_id=active_run.info.run_id,
            )
