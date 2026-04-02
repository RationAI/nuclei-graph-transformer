import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyvips
import torch
from einops import rearrange
from lightning import Callback, LightningModule, Trainer
from openslide import OpenSlide
from PIL import Image as PILImage
from PIL import ImageDraw
from rationai.masks import slide_resolution, write_big_tiff

from nuclei_graph.nuclei_graph_typing import Outputs, PredictBatch


class WSLPredictionMasksCallback(Callback):
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
        outputs: Outputs,
        batch: PredictBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch["metadata"][0]  # batch size is 1

        # get scale factors for converting polygon coordinates to mask pixel coordinates
        with OpenSlide(Path(metadata["slide_path"])) as slide:
            mask_size = slide.level_dimensions[self.level]
            base_mpp_x, base_mpp_y = slide_resolution(slide, 0)
            mask_mpp_x, mask_mpp_y = slide_resolution(slide, self.level)
            scale_x = base_mpp_x / mask_mpp_x
            scale_y = base_mpp_y / mask_mpp_y

        mask = PILImage.new("L", mask_size, color=0)
        canvas = ImageDraw.Draw(mask)

        # extract and align predictions
        logits_permuted = outputs["nuclei"][0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        logits = logits_permuted[:seq_len][metadata["perm_inverse"]]

        preds = torch.sigmoid(logits).cpu().numpy().flatten()

        nuclei_path = metadata["slide_nuclei_path"]
        nuclei_df = pd.read_parquet(nuclei_path, columns=["id", "polygon"])
        nuclei_df = nuclei_df.sort_values("id").reset_index(drop=True)
        polygons = nuclei_df["polygon"].values

        # draw polygon masks
        for poly, pred in zip(polygons, preds, strict=True):
            polygon = rearrange(poly, "(n c) -> n c", c=2)
            scaled_poly = [(x * scale_x, y * scale_y) for x, y in polygon]
            pixel_val = int(pred * 255)
            canvas.polygon(scaled_poly, fill=pixel_val, outline=pixel_val)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / f"{metadata['slide_id']}.tiff"

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


class MILAttentionMasksCallback(Callback):
    def __init__(
        self,
        level: int,
        mask_tile_width: int = 512,
        mask_tile_height: int = 512,
        mlflow_artifact_path: str = "attention_masks",
    ) -> None:
        """Callback to generate and log MIL attention masks at a desired `level`."""
        super().__init__()
        self.level = level
        self.mask_tile_width = mask_tile_width
        self.mask_tile_height = mask_tile_height
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

        # get scale factors for converting polygon coordinates to mask pixel coordinates
        with OpenSlide(Path(metadata["slide_path"])) as slide:
            mask_size = slide.level_dimensions[self.level]
            base_mpp_x, base_mpp_y = slide_resolution(slide, 0)
            mask_mpp_x, mask_mpp_y = slide_resolution(slide, self.level)
            scale_x = base_mpp_x / mask_mpp_x
            scale_y = base_mpp_y / mask_mpp_y

        mask = PILImage.new("L", mask_size, color=0)
        canvas = ImageDraw.Draw(mask)

        # extract and align attention scores
        attn_permuted = outputs["attn_weights"][0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        attn_scores = attn_permuted[:seq_len][metadata["perm_inverse"]]
        attn_scores = attn_scores.cpu().numpy().flatten()

        max_score = attn_scores.max()
        if max_score > 0:
            attn_scores = attn_scores / max_score

        nuclei_path = metadata["slide_nuclei_path"]
        nuclei_df = pd.read_parquet(nuclei_path, columns=["id", "polygon"])
        nuclei_df = nuclei_df.sort_values("id").reset_index(drop=True)
        polygons = nuclei_df["polygon"].values

        # draw polygon masks
        for poly, pred in zip(polygons, attn_scores, strict=True):
            polygon = rearrange(poly, "(n c) -> n c", c=2)
            scaled_poly = [(x * scale_x, y * scale_y) for x, y in polygon]
            pixel_val = int(pred * 255)
            canvas.polygon(scaled_poly, fill=pixel_val, outline=pixel_val)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / f"{metadata['slide_id']}.tiff"

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
