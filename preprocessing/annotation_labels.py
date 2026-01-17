"""Script for nuclei labeling based on annotation masks.

Assumes the following structure of input data:
1. Segmented nuclei (`preprocessing/nuclei_segmentation.py`):
<NUCLEI_PATH>/
    <DATASET_NAME>/
        slide_id=<SLIDE_NAME>/
            *.parquet (columns "id" (str), "polygon" (np.ndarray[float]) and "centroid" (np.ndarray[float]))

2. Annotation masks (`preprocessing/annotation_masks.py`):
<ANNOT_MASKS_URI>/
    <SLIDE_NAME>.tiff (binary single-channel mask of annotated regions)

The labels are stored only for positive slides with annotations. Each nucleus is assigned label 1
if the fraction of its polygon vertices falling inside the mask ≥ `overlap_threshold`, otherwise 0.

The result is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.parquet (columns "slide_id" (str), "id" (str), and "annot_label" (int))
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
import ray
import tifffile
from einops import rearrange
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks.processing import process_items
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote
def label_slide(
    slide_path: Path,
    nuclei_dir: Path,
    annot_masks_dir: Path,
    output_dir: Path,
    overlap_thr: float = 1.0,
) -> None:
    dataset_name = slide_path.parents[0].name
    nuclei_path = nuclei_dir / dataset_name / f"slide_id={slide_path.stem}"
    nuclei = pd.read_parquet(nuclei_path, columns=["id", "polygon"]).sort_values("id")
    nuclei["slide_id"] = slide_path.stem

    annot_mask_path = annot_masks_dir / f"{slide_path.stem}.tiff"
    annot_mask: NDArray[np.uint8] = tifffile.imread(annot_mask_path).squeeze()

    mask_extent_y, mask_extent_x = annot_mask.shape  # assumes mask is single-channel
    with OpenSlide(slide_path) as slide:
        wsi_extent_x, wsi_extent_y = slide.dimensions
    scale_x = mask_extent_x / wsi_extent_x
    scale_y = mask_extent_y / wsi_extent_y

    # compute the fraction of polygon vertices falling inside the annotation mask
    polygons = rearrange(nuclei["polygon"].tolist(), "b (v d) -> b v d", d=2)
    coords = np.round(polygons * np.array([scale_x, scale_y])).astype(int)
    x_coords = np.clip(coords[..., 0], 0, mask_extent_x - 1)
    y_coords = np.clip(coords[..., 1], 0, mask_extent_y - 1)
    coverage = np.mean(annot_mask[y_coords, x_coords] != 0, axis=1)

    nuclei["annot_label"] = (coverage >= overlap_thr).astype(int)

    output_path = output_dir / f"{slide_path.stem}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nuclei[["slide_id", "id", "annot_label"]].to_parquet(output_path, index=False)


@hydra.main(
    config_path="../configs",
    config_name="preprocessing/annotation_labels",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    annot_masks_dir = Path(download_artifacts(config.annot_masks_uri))
    train_slides = pd.read_csv(download_artifacts(config.train_metadata_uri))
    test_slides = pd.read_csv(download_artifacts(config.test_metadata_uri))
    slides = pd.concat([train_slides, test_slides])
    slides_annotated = slides[slides["is_carcinoma"] & slides["has_annotation"]]

    with TemporaryDirectory() as tmp_dir:
        process_items(
            items=slides_annotated["slide_path"].map(Path),
            process_item=label_slide,
            fn_kwargs={
                "nuclei_dir": Path(config.nuclei_path),
                "annot_masks_dir": annot_masks_dir,
                "output_dir": Path(tmp_dir),
                "overlap_thr": config.overlap_threshold,
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(
            local_dir=tmp_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
