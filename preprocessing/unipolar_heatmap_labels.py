"""Script for unipolar heatmap-based nuclei labeling.

This script can be used for labeling nuclei according to any unipolar mask (e.g., binary annotation masks,
model prediction heatmaps, etc.). Each nucleus is assigned label 1 if the fraction of its polygon vertices
falling inside the thresholded heatmap (intensity > `positive_threshold`) is ≥ `overlap_threshold`, otherwise 0.
The labels are stored only for positive slides.

Assumes the following structure of input data:
1. Segmented nuclei (`preprocessing/nuclei_segmentation.py`):
<NUCLEI_PATH>/
    <DATASET_NAME>/
        slide_id=<SLIDE_NAME>/
            *.parquet (columns "id" (str), "polygon" (np.ndarray[float]) and "centroid" (np.ndarray[float]))

2. Metadatasets for processing (`metadata_mapping/...`):
[ <SLIDES_METADATA_URI>.parquet (columns "slide_id" (str), "slide_nuclei_path" (str), "slide_path" (str) and "is_carcinoma" (bool)), ... ]

3. Heatmaps or binary masks (e.g. `preprocessing/annotation_masks.py`):
<HEATMAPS_URI>/
    <SLIDE_NAME>.tiff (unipolar heatmap with intensities in [0, 255])

The result is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.parquet (columns "slide_id" (str), "id" (str), and <LABEL_COLUMN> (int))
"""

from pathlib import Path
from typing import Any

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
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(num_cpus=1, memory=(3 * 1024**3))
def label_slide(
    metadata: dict[str, Any],
    heatmaps_dir: Path,
    output_dir: Path,
    label_column: str,
    overlap_thr: float,
    positive_thr: float,
) -> None:
    nuclei = pd.read_parquet(metadata["slide_nuclei_path"], columns=["id", "polygon"])
    nuclei = nuclei.sort_values("id").reset_index(drop=True)
    nuclei["slide_id"] = metadata["slide_id"]

    mask_path = heatmaps_dir / f"{metadata['slide_id']}.tiff"
    mask: NDArray[np.uint8] = tifffile.imread(mask_path).squeeze()

    mask_extent_y, mask_extent_x = mask.shape  # assumes mask is single-channel
    with OpenSlide(metadata["slide_path"]) as slide:
        wsi_extent_x, wsi_extent_y = slide.dimensions
    scale_x = mask_extent_x / wsi_extent_x
    scale_y = mask_extent_y / wsi_extent_y

    # compute the fraction of polygon vertices falling inside the mask
    polygons = rearrange(nuclei["polygon"].tolist(), "b (v d) -> b v d", d=2)
    coords = np.round(polygons * np.array([scale_x, scale_y])).astype(int)
    x_coords = np.clip(coords[..., 0], 0, mask_extent_x - 1)
    y_coords = np.clip(coords[..., 1], 0, mask_extent_y - 1)
    coverage = np.mean(mask[y_coords, x_coords] / 255.0 > positive_thr, axis=1)

    nuclei[label_column] = (coverage >= overlap_thr).astype(int)

    output_path = output_dir / f"{metadata['slide_id']}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nuclei[["slide_id", "id", label_column]].to_parquet(output_path, index=False)


def uris2df(uris: list[str] | None, cols: list[str]) -> pd.DataFrame:
    """Loads and merges multiple metadata .parquet files into a single DataFrame."""
    if not uris:
        return pd.DataFrame(columns=cols)
    batches = [
        pd.read_parquet(Path(download_artifacts(uri)), columns=cols) for uri in uris
    ]
    return pd.concat(batches, ignore_index=True).drop_duplicates(subset=["slide_path"])


@with_cli_args(["+preprocessing=unipolar_heatmap_labels"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, _: MLFlowLogger) -> None:
    heatmaps_dir = download_artifacts(config.heatmap_uri)
    cols = ["slide_id", "slide_path", "slide_nuclei_path"]
    slides = uris2df(config.metadata_uris, [*cols, "is_carcinoma"])
    slides_carcinoma = slides[slides["is_carcinoma"]]
    to_process = slides_carcinoma[cols]

    process_items(
        items=to_process.to_dict("records"),
        process_item=label_slide,
        fn_kwargs={
            "heatmaps_dir": Path(heatmaps_dir),
            "output_dir": Path(config.output_path),
            "label_column": config.label_column,
            "overlap_thr": config.overlap_threshold,
            "positive_thr": config.positive_threshold,
        },
        max_concurrent=config.max_concurrent,
    )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
