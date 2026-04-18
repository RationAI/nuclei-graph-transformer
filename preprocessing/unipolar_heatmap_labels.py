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

2. Metadatasets for processing (`exploration/save_metadataset.py`):
[ <SLIDES_METADATA_URI>.csv (columns "slide_path" (str) and "is_carcinoma" (bool)), ...]

3. (Optional) Exclusion CSVs logged in MLflow (`preprocessing/annotation_masks.py`):
[ <MISSING_HEATMAPS_URI>.csv (column "slide_path" (str)), ... ]

4. Heatmaps or binary masks (`preprocessing/annotation_masks.py`):
<HEATMAPS_URI>/
    <SLIDE_NAME>.tiff (unipolar heatmap with intensities in [0, 255])

The result is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.parquet (columns "slide_id" (str), "id" (str), and <LABEL_COLUMN> (int))
"""

from pathlib import Path

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
    slide_path: Path,
    nuclei_dir: Path,
    heatmaps_dir: Path,
    output_dir: Path,
    label_column: str,
    overlap_thr: float,
    positive_thr: float,
) -> None:
    dataset_name = slide_path.parents[0].name
    nuclei_path = nuclei_dir / dataset_name / f"slide_id={slide_path.stem}"
    nuclei = pd.read_parquet(nuclei_path, columns=["id", "polygon"]).sort_values("id")
    nuclei["slide_id"] = slide_path.stem

    mask_path = heatmaps_dir / f"{slide_path.stem}.tiff"
    mask: NDArray[np.uint8] = tifffile.imread(mask_path).squeeze()

    mask_extent_y, mask_extent_x = mask.shape  # assumes mask is single-channel
    with OpenSlide(slide_path) as slide:
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

    output_path = output_dir / f"{slide_path.stem}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nuclei[["slide_id", "id", label_column]].to_parquet(output_path, index=False)


def uris2df(uris: list[str] | None) -> pd.DataFrame:
    """Loads and merges multiple metadata CSVs into a single DataFrame."""
    if not uris:
        return pd.DataFrame(columns=["slide_path"])
    batches = [pd.read_csv(Path(download_artifacts(uri))) for uri in uris]
    return pd.concat(batches, ignore_index=True).drop_duplicates(subset=["slide_path"])


@with_cli_args(["+preprocessing=unipolar_heatmap_labels"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, _: MLFlowLogger) -> None:
    heatmaps_dir = download_artifacts(config.heatmap_uri)
    slides = uris2df(config.metadata_uris)
    exclude_slides = uris2df(config.exclude_slides_uris)
    valid_mask = (slides["is_carcinoma"]) & (
        ~slides["slide_path"].isin(exclude_slides["slide_path"])
    )

    process_items(
        items=slides.loc[valid_mask, "slide_path"].map(Path),
        process_item=label_slide,
        fn_kwargs={
            "nuclei_dir": Path(config.nuclei_path),
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
