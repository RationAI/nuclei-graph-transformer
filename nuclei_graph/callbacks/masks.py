import tempfile
from pathlib import Path
from typing import Any

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
from rationai.masks.mask_builders import ScalarMaskBuilder

from nuclei_graph.nuclei_graph_typing import Batch, Outputs


class BaseMasksCallback(Callback):
    def __init__(
        self,
        level: int = 2,
        mask_tile_width: int = 512,
        mask_tile_height: int = 512,
        mlflow_artifact_path: str = "masks",
    ) -> None:
        super().__init__()
        self.level = level
        self.mask_tile_width = mask_tile_width
        self.mask_tile_height = mask_tile_height
        self.mlflow_artifact_path = mlflow_artifact_path
        self.tmp_dir: tempfile.TemporaryDirectory[str] | None = None

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


class TileHeatmapMasksCallback(BaseMasksCallback):
    """Generates probability heatmaps for tile predictions."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mlflow_artifact_path", "tile_heatmaps")
        super().__init__(**kwargs)
        self.mask_builders: dict[str, ScalarMaskBuilder] = {}

    def _get_dataset(self, trainer: Trainer):
        """Safely extracts the predict dataset from Lightning internals."""
        if hasattr(trainer, "predict_dataloaders") and hasattr(
            trainer.predict_dataloaders, "dataset"
        ):
            return trainer.predict_dataloaders.dataset
        return trainer.datamodule.predict_dataset

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits_graph = outputs["graph"].view(-1)
        preds_graph = torch.sigmoid(logits_graph).cpu()

        metadata = batch["metadata"]
        assert metadata is not None, "Metadata is required"
        slide_ids = metadata["slide_id"]
        xs = metadata["x"]
        ys = metadata["y"]

        dataset = self._get_dataset(trainer)
        unique_slides = set(slide_ids)

        for slide_id in unique_slides:
            if slide_id not in self.mask_builders:
                slide_row = dataset.slides[dataset.slides["stem"] == slide_id].iloc[0]

                if "mpp_x" in slide_row:
                    mpp_x, mpp_y = slide_row["mpp_x"], slide_row["mpp_y"]
                else:
                    meta_row = dataset.metadata.loc[slide_id]
                    mpp_x, mpp_y = meta_row["mpp_x"], meta_row["mpp_y"]

                extent_tile = slide_row["tile_extent_x"]
                stride = slide_row.get("stride_x", extent_tile)

                self.mask_builders[slide_id] = ScalarMaskBuilder(
                    save_dir=Path(self.tmp_dir.name),
                    filename=str(slide_id),
                    extent_x=int(slide_row["extent_x"]),
                    extent_y=int(slide_row["extent_y"]),
                    mpp_x=float(mpp_x),
                    mpp_y=float(mpp_y),
                    extent_tile=int(extent_tile),
                    stride=int(stride),
                    device="cpu",
                )

            indices = [i for i, sid in enumerate(slide_ids) if sid == slide_id]

            slide_preds = preds_graph[indices].unsqueeze(-1)
            slide_xs = torch.tensor([xs[i] for i in indices], dtype=torch.float32)
            slide_ys = torch.tensor([ys[i] for i in indices], dtype=torch.float32)

            self.mask_builders[slide_id].update(slide_preds, slide_xs, slide_ys)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.tmp_dir is not None:
            for builder in self.mask_builders.values():
                builder.save()
            self.mask_builders.clear()
        super().on_predict_epoch_end(trainer, pl_module)


class NucleiPredictionMasksCallback(BaseMasksCallback):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mlflow_artifact_path", "prediction_masks")
        super().__init__(**kwargs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch["metadata"]
        assert metadata is not None, "Metadata is required to save predictions."

        # Batch Size is 1
        slide_id = metadata["slide_id"][0]
        slide_path = metadata["slide_path"][0]
        nuclei_ids = metadata["nuclei_ids"][0]
        nuclei_path = metadata["slide_nuclei_path"][0]

        # get scale factors for converting polygon coordinates to mask pixel coordinates
        with OpenSlide(Path(slide_path)) as slide:
            mask_size = slide.level_dimensions[self.level]
            base_mpp_x, base_mpp_y = slide_resolution(slide, 0)
            mask_mpp_x, mask_mpp_y = slide_resolution(slide, self.level)
            scale_x = base_mpp_x / mask_mpp_x
            scale_y = base_mpp_y / mask_mpp_y

        mask = PILImage.new("L", mask_size, color=0)
        canvas = ImageDraw.Draw(mask)

        logits = outputs["nuclei"].squeeze(-1)
        preds_t = torch.sigmoid(logits).cpu().numpy().flatten()

        # Map predictions to IDs and sort to restore original file order
        preds_df = (
            pd.DataFrame({"id": nuclei_ids, "prediction": preds_t})
            .sort_values("id")
            .reset_index(drop=True)
        )

        aligned_preds = preds_df["prediction"].values

        nuclei_df = pd.read_parquet(nuclei_path, columns=["id", "polygon"])
        nuclei_df = nuclei_df.sort_values("id").reset_index(drop=True)
        polygons = nuclei_df["polygon"].values

        # draw polygon masks
        for poly, pred in zip(polygons, aligned_preds, strict=True):
            polygon = rearrange(poly, "(n c) -> n c", c=2)
            scaled_poly = [(x * scale_x, y * scale_y) for x, y in polygon]
            pixel_val = int(pred * 255)
            canvas.polygon(scaled_poly, fill=pixel_val, outline=pixel_val)

        output_path = self._get_output_path(slide_id)

        write_big_tiff(
            image=pyvips.Image.new_from_array(np.array(mask)),
            path=output_path,
            mpp_x=mask_mpp_x,
            mpp_y=mask_mpp_y,
            tile_width=self.mask_tile_width,
            tile_height=self.mask_tile_height,
        )


class AttentionMasksCallback(BaseMasksCallback):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mlflow_artifact_path", "attention_masks")
        super().__init__(**kwargs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch["metadata"]
        assert metadata is not None, "Metadata is required to save predictions."

        # Batch Size is 1
        slide_id = metadata["slide_id"][0]
        slide_path = metadata["slide_path"][0]
        nuclei_ids = metadata["nuclei_ids"][0]
        nuclei_path = metadata["slide_nuclei_path"][0]

        # get scale factors for converting polygon coordinates to mask pixel coordinates
        with OpenSlide(Path(slide_path)) as slide:
            mask_size = slide.level_dimensions[self.level]
            base_mpp_x, base_mpp_y = slide_resolution(slide, 0)
            mask_mpp_x, mask_mpp_y = slide_resolution(slide, self.level)
            scale_x = base_mpp_x / mask_mpp_x
            scale_y = base_mpp_y / mask_mpp_y

        mask = PILImage.new("L", mask_size, color=0)
        canvas = ImageDraw.Draw(mask)

        attn = outputs["attn_weights"].squeeze(-1)
        attn_scores_raw = attn.cpu().numpy().flatten()

        attn_df = (
            pd.DataFrame({"id": nuclei_ids, "score": attn_scores_raw})
            .sort_values("id")
            .reset_index(drop=True)
        )

        attn_scores = np.asarray(attn_df["score"].values, dtype=np.float32)

        max_score = float(attn_scores.max())
        if max_score > 0:
            attn_scores = attn_scores / max_score

        nuclei_df = pd.read_parquet(nuclei_path, columns=["id", "polygon"])
        nuclei_df = nuclei_df.sort_values("id").reset_index(drop=True)
        polygons = nuclei_df["polygon"].values

        # draw polygon masks
        for poly, pred in zip(polygons, attn_scores, strict=True):
            polygon = rearrange(poly, "(n c) -> n c", c=2)
            scaled_poly = [(x * scale_x, y * scale_y) for x, y in polygon]
            pixel_val = int(pred * 255)
            canvas.polygon(scaled_poly, fill=pixel_val, outline=pixel_val)

        output_path = self._get_output_path(slide_id)

        write_big_tiff(
            image=pyvips.Image.new_from_array(np.array(mask)),
            path=output_path,
            mpp_x=mask_mpp_x,
            mpp_y=mask_mpp_y,
            tile_width=self.mask_tile_width,
            tile_height=self.mask_tile_height,
        )
