import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyvips
import torch
from einops import rearrange
from lightning import Callback, LightningModule, Trainer
from nuclei_graph.nuclei_graph_typing import PredictBatch
from openslide import OpenSlide
from PIL import Image as PILImage
from PIL import ImageDraw
from rationai.masks import slide_resolution, write_big_tiff
from scipy.interpolate import NearestNDInterpolator
from torch import Tensor


class PredictionMasksCallback(Callback):
    def __init__(
        self,
        level: int,
        mask_tile_width: int = 512,
        mask_tile_height: int = 512,
        mlflow_artifact_path: str = "prediction_masks",
    ) -> None:
        """Callback to generate and log nuclei prediction masks at a desired `level`."""
        super().__init__()
        self.level = level
        self.mask_tile_width = mask_tile_width
        self.mask_tile_height = mask_tile_height
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
        slide_id = metadata["slide_id"]
        nuclei_path = Path(metadata["slide_nuclei_path"])
        slide_path = Path(metadata["slide_path"])

        nuclei = pd.read_parquet(nuclei_path, columns=["id", "centroid", "polygon"])
        nuclei = nuclei.sort_values("id").reset_index(drop=True)

        # extract and align predictions
        logits = outputs[0].squeeze(-1)  # (n,)
        keep_indices = metadata["keep_indices"]
        logits_unpadded = logits[: len(keep_indices)]
        logits_ordered = logits_unpadded[metadata["perm_inverse"]]

        predicted_labels = torch.sigmoid(logits_ordered).cpu().numpy().flatten()
        nuclei.loc[keep_indices.cpu().numpy(), "prediction"] = predicted_labels

        # interpolate missing nuclei
        if nuclei["prediction"].isna().any():
            valid = nuclei.dropna(subset=["prediction"])
            coords = np.stack(valid["centroid"].tolist())
            interp = NearestNDInterpolator(coords, valid["prediction"].values)

            missing_mask = nuclei["prediction"].isna()
            missing_coords = np.stack(nuclei.loc[missing_mask, "centroid"].tolist())
            nuclei.loc[missing_mask, "prediction"] = interp(missing_coords)

        with OpenSlide(slide_path) as slide:
            mask_size = slide.level_dimensions[self.level]
            base_mpp_x, base_mpp_y = slide_resolution(slide, 0)
            mask_mpp_x, mask_mpp_y = slide_resolution(slide, self.level)
            scale_x = base_mpp_x / mask_mpp_x
            scale_y = base_mpp_y / mask_mpp_y

        mask = PILImage.new("L", mask_size, color=0)
        canvas = ImageDraw.Draw(mask)

        # draw prediction heatmaps
        for _, row in nuclei.iterrows():
            poly = rearrange(row["polygon"], "(n c) -> n c", c=2)
            scaled_poly = [(x * scale_x, y * scale_y) for x, y in poly]
            pixel_val = int(row["prediction"] * 255)
            canvas.polygon(scaled_poly, fill=pixel_val, outline=pixel_val)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / f"{slide_id}.tiff"

            write_big_tiff(
                image=pyvips.Image.new_from_array(np.array(mask)),
                path=output_path,
                mpp_x=mask_mpp_x,
                mpp_y=mask_mpp_y,
                tile_width=self.mask_tile_width,
                tile_height=self.mask_tile_height,
            )

            active_run = mlflow.active_run()
            assert active_run is not None
            mlflow.log_artifact(
                local_path=str(output_path),
                artifact_path=self.mlflow_artifact_path,
                run_id=active_run.info.run_id,
            )
