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


class BaseMasksCallback(Callback):
    def __init__(
        self,
        level: int,
        mask_tile_width: int = 512,
        mask_tile_height: int = 512,
        mlflow_artifact_path: str = "masks",
    ) -> None:
        super().__init__()
        self.level = level
        self.mask_tile_width = mask_tile_width
        self.mask_tile_height = mask_tile_height
        self.mlflow_artifact_path = mlflow_artifact_path
        self.tmp_dir = None

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

    def _get_output_path(self, slide_id: str) -> Path:
        assert self.tmp_dir is not None
        return Path(self.tmp_dir.name) / f"{slide_id}.tiff"

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


class WSLPredictionMasksCallback(BaseMasksCallback):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("mlflow_artifact_path", "prediction_masks")
        super().__init__(**kwargs)

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
        logits = outputs["nuclei"][0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        logits_unpadded = logits[:seq_len]
        logits_ordered = logits_unpadded[metadata["perm_inverse"]]

        predicted_labels = torch.sigmoid(logits_ordered).cpu().numpy().flatten()

        nuclei_path = metadata["slide_nuclei_path"]
        nuclei_df = pd.read_parquet(nuclei_path, columns=["id", "polygon"])
        nuclei_df = nuclei_df.sort_values("id").reset_index(drop=True)
        polygons = nuclei_df["polygon"].values

        # draw polygon masks
        for poly, pred in zip(polygons, predicted_labels, strict=True):
            polygon = rearrange(poly, "(n c) -> n c", c=2)
            scaled_poly = [(x * scale_x, y * scale_y) for x, y in polygon]
            pixel_val = int(pred * 255)
            canvas.polygon(scaled_poly, fill=pixel_val, outline=pixel_val)

        output_path = self._get_output_path(metadata["slide_id"])

        write_big_tiff(
            image=pyvips.Image.new_from_array(np.array(mask)),
            path=output_path,
            mpp_x=mask_mpp_x,
            mpp_y=mask_mpp_y,
            tile_width=self.mask_tile_width,
            tile_height=self.mask_tile_height,
        )


class MILAttentionMasksCallback(BaseMasksCallback):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("mlflow_artifact_path", "attention_masks")
        super().__init__(**kwargs)

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
        attn = outputs["attn_weights"][0].squeeze(-1)  # (n,)
        seq_len = batch["slides"]["seq_len"][0].item()
        attn_unpadded = attn[:seq_len]
        attn_ordered = attn_unpadded[metadata["perm_inverse"]]

        attn_scores = attn_ordered.cpu().numpy().flatten()

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

        output_path = self._get_output_path(metadata["slide_id"])

        write_big_tiff(
            image=pyvips.Image.new_from_array(np.array(mask)),
            path=output_path,
            mpp_x=mask_mpp_x,
            mpp_y=mask_mpp_y,
            tile_width=self.mask_tile_width,
            tile_height=self.mask_tile_height,
        )
