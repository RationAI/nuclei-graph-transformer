"""Script for annotation-based nuclei labeling on the PANDA dataset.

Each nucleus is assigned a binary label (1/0) indicating if the fraction of its vertices and
the annotation (annotation label >= 3 for Radboud and >= 2 for Karolinska) is >= `overlap_threshold`.
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
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(num_cpus=1, memory=(3 * 1024**3))
def label_slide(
    metadata: dict,
    nuclei_dir: Path,
    annots_dir: Path,
    output_dir: Path,
    overlap_thr: float,
) -> None:
    slide_id = metadata["slide_id"]
    wsi_extent_x, wsi_extent_y = metadata["extent_x"], metadata["extent_y"]
    provider = metadata["data_provider"]

    nuclei_path = nuclei_dir / f"slide_id={slide_id}"
    nuclei = pd.read_parquet(nuclei_path, columns=["id", "polygon"]).sort_values("id")
    nuclei["slide_id"] = slide_id

    mask_path = annots_dir / f"{slide_id}_mask.tiff"
    mask: NDArray[np.uint8] = tifffile.imread(mask_path)
    if mask.ndim == 3:  # values are repeated across channels, take the first one
        mask = mask[..., 0] if mask.shape[-1] == 3 else mask.squeeze()
    else:
        mask = mask.squeeze()
    mask_extent_y, mask_extent_x = mask.shape

    scale_x = mask_extent_x / wsi_extent_x
    scale_y = mask_extent_y / wsi_extent_y

    polygons = rearrange(nuclei["polygon"].tolist(), "b (v d) -> b v d", d=2)
    coords = np.round(polygons * np.array([scale_x, scale_y])).astype(int)
    x_coords = np.clip(coords[..., 0], 0, mask_extent_x - 1)
    y_coords = np.clip(coords[..., 1], 0, mask_extent_y - 1)

    annot_labels = mask[y_coords, x_coords]

    if provider == "radboud":
        is_carcinoma_vertex = annot_labels >= 3
    else:  # karolinska
        is_carcinoma_vertex = annot_labels >= 2

    coverage = np.mean(is_carcinoma_vertex, axis=1)
    nuclei["annot_label"] = (coverage >= overlap_thr).astype(int)

    output_path = output_dir / f"{slide_id}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nuclei[["slide_id", "id", "annot_label"]].to_parquet(output_path, index=False)


@with_cli_args(["+preprocessing=annotation_labels"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = pd.read_csv(Path(download_artifacts(config.metadata_uri)))
    to_process = slides[slides["is_carcinoma"] & slides["annotation"]]
    to_process = to_process[["slide_id", "data_provider", "extent_x", "extent_y"]]

    with TemporaryDirectory() as tmp_dir:
        process_items(
            items=to_process.to_dict("records"),
            process_item=label_slide,
            fn_kwargs={
                "nuclei_dir": Path(config.nuclei_path),
                "annots_dir": Path(config.train_label_masks),
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
