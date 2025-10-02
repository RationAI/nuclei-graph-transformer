"""Lightning callback to save nuclei .tiff predictions for visualisation purposes."""

import os
import tempfile

import mlflow
import pyvips
import torch
from lightning import Callback, LightningModule, Trainer
from openslide import OpenSlide
from rationai.masks import slide_resolution
from torch import Tensor

from nuclei_graph.masks import NucleiMask
from nuclei_graph.typing import Metadata, Outputs, PredictInput


LEVEL = 1
BASE_LEVEL = 0


class NucleiHeatmapCallback(Callback):
    def save_mask(self, logits: Tensor, metadata: Metadata) -> None:
        slide_id = metadata["slide_id"]
        slide_tiff_path = metadata["slide_tiff_path"]
        raw_cells_path = metadata["raw_cells_path"]
        nuclei_count = metadata["nuclei_count"]
        perm_inverse = metadata["perm_inverse"]

        logits_unpadded = logits[:nuclei_count]
        logits_original_order = logits_unpadded[perm_inverse]
        predicted_labels = torch.sigmoid(logits_original_order).squeeze()

        with OpenSlide(slide_tiff_path) as slide:
            base_mpp_x, base_mpp_y = slide_resolution(slide, level=BASE_LEVEL)
            mask_mpp_x, mask_mpp_y = slide_resolution(slide, level=LEVEL)
            annotator = NucleiMask(
                base_mpp_x=base_mpp_x,
                base_mpp_y=base_mpp_y,
                mask_size=slide.level_dimensions[LEVEL],
                mask_mpp_x=mask_mpp_x,
                mask_mpp_y=mask_mpp_y,
                labels=predicted_labels,
                raw_cells_path=str(raw_cells_path),
            )

        mask = annotator()
        vips_img = pyvips.Image.new_from_array(mask)

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
            file_path = f"{temp_dir}/{slide_id}.tiff"
            xres = 1000 / annotator.mask_mpp_x
            yres = 1000 / annotator.mask_mpp_y
            # needed for compatability in xOpat
            vips_img.tiffsave(
                file_path,
                bigtiff=True,
                compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
                tile=True,
                tile_width=256,
                tile_height=256,
                xres=xres,
                yres=yres,
                pyramid=True,
            )
            active_run = mlflow.active_run()
            assert active_run is not None

            mlflow.log_artifacts(
                temp_dir,
                "predictions",
                run_id=active_run.info.run_id,
            )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: PredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, metadata = batch
        self.save_mask(outputs[0], metadata[0])  # batch size is one
