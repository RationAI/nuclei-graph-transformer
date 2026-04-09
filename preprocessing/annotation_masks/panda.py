"""Script for generating prostate cancer annotation masks for the PANDA Challenge Dataset.

Label Systems
- Radboud: individually labeled prostate glands
    0 	Background, non-tissue, or unknown
    1 	Stroma, connective tissue, non-epithelial tissue
    2 	Healthy (benign) epithelium
    3 	Cancerous epithelium (Gleason 3)
    4 	Cancerous epithelium (Gleason 4)
    5 	Cancerous epithelium (Gleason 5)
- Karolinska: labeled regions
    0 	Background, non-tissue, or unknown
    1 	Benign tissue (stroma and epithelium combined)
    2 	Cancerous tissue (stroma and epithelium combined)

This script generates binary masks of carcinoma regions (carcinoma = labels 3, 4, and 5
for Radboud, label 2 for Karolinska).
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pyvips
import ray
import tifffile
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.masks import write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


@ray.remote(num_cpus=1, memory=(5 * 1024**3))
def process_slide(
    metadata: dict[str, Any],
    annots_dir: Path,
    output_dir: str,
    mask_tile_width: int,
    mask_tile_height: int,
) -> None:
    mask_path = annots_dir / f"{metadata['slide_id']}_mask.tiff"
    with OpenSlide(mask_path) as slide:
        mpp_x, mpp_y = slide.slide_resolution(level=0)

    mask: NDArray[np.uint8] = tifffile.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]

    provider = metadata["data_provider"]
    threshold = 3 if provider == "radboud" else 2
    binary_mask = (mask >= threshold).astype(np.uint8) * 255

    output_path = Path(output_dir) / provider / f"{metadata['slide_id']}.tiff"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_big_tiff(
        image=pyvips.Image.new_from_array(binary_mask),
        path=output_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        tile_width=mask_tile_width,
        tile_height=mask_tile_height,
    )


@with_cli_args(["+preprocessing=annotation_masks/panda"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = pd.read_csv(download_artifacts(config.metadata_uri))
    slides_annots = slides[slides["has_annotation"]]
    to_process = slides_annots[["slide_id", "data_provider"]]

    with TemporaryDirectory(dir=os.getcwd()) as output_dir:
        process_items(
            items=to_process.to_dict("records"),
            process_item=process_slide,
            fn_kwargs={
                "annots_dir": Path(config.annotations_dir),
                "output_dir": output_dir,
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(local_dir=output_dir)


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
