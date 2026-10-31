"""Script for precomputing the nuclei texture patches of the `bbox` embedding mode.

The patches depend only on the slide pixels and the true (non-augmented) nuclei
geometry, so they are extracted once here instead of in every training step. For
each slide, it stores in `output_path/slide_id=<id>/`:

- `patches.npy`: (n, 3, patch_size, patch_size) uint8 patches, in a spatial
  (k-d tree) order, so a spatially contiguous crop reads mostly contiguous rows,
- `rows.npy`: (n,) int32 row in `patches.npy` of each nucleus, in the order of
  `BaseCropDataset.get_nuclei` (sorted by `id`).

The patch geometry is written to `output_path/params.json`, which the datasets
check against their own when reading the patches.
"""

import json
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import ray
from einops import rearrange
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.data.block_mask import block_spatial_sort
from nuclei_graph.data.datasets.nuclei_features import (
    PARAMS_FILE,
    PATCHES_FILE,
    ROWS_FILE,
    NucleiFeatureExtractor,
    patches_params,
)


@ray.remote(num_cpus=1, memory=(4 * 1024**3))
def extract_patches(
    item: dict[str, Any], output_dir: Path, patch_size: int, leaf_size: int
) -> None:
    slide_dir = output_dir / f"slide_id={item['slide_id']}"
    if (slide_dir / ROWS_FILE).exists():  # written last
        return

    nuclei = pd.read_parquet(
        item["slide_nuclei_path"], columns=["id", "polygon", "centroid"]
    )
    nuclei = nuclei.sort_values("id").reset_index(drop=True)
    polygons = rearrange(nuclei["polygon"].tolist(), "n (v c) -> n v c", c=2)
    centroids = np.stack(nuclei["centroid"].tolist())

    extractor = NucleiFeatureExtractor()
    extractor.patch_size = patch_size
    patches = extractor.get_nuclei_bboxes(
        centroids.astype(np.float32),
        polygons.astype(np.float32),
        item["slide_path"],
        item["mpp_x"],
        item["mpp_y"],
    )
    assert patches is not None

    order = block_spatial_sort(centroids, leaf_size)
    rows = np.empty(len(order), dtype=np.int32)
    rows[order] = np.arange(len(order), dtype=np.int32)

    slide_dir.mkdir(parents=True, exist_ok=True)
    for name, array in ((PATCHES_FILE, patches.numpy()[order]), (ROWS_FILE, rows)):
        tmp_path = slide_dir / f".{name}.tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, array)
        os.replace(tmp_path, slide_dir / name)


@with_cli_args(["+preprocessing=nuclei_patches"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, _: MLFlowLogger) -> None:
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = patches_params(config.patch_size)
    params_path = output_dir / PARAMS_FILE
    if params_path.exists():
        existing = json.loads(params_path.read_text())
        assert existing == params, (
            f"{output_dir} holds patches with {existing}, not {params}; "
            "choose another `output_path`."
        )
    else:
        params_path.write_text(json.dumps(params, indent=2))

    metadata = pd.concat(
        [pd.read_parquet(download_artifacts(uri)) for uri in config.metadata_uris]
    ).drop_duplicates("slide_id")

    process_items(
        items=metadata[
            ["slide_id", "slide_path", "slide_nuclei_path", "mpp_x", "mpp_y"]
        ].to_dict("records"),
        process_item=extract_patches,
        fn_kwargs={
            "output_dir": output_dir,
            "patch_size": config.patch_size,
            "leaf_size": config.leaf_size,
        },
        max_concurrent=config.max_concurrent,
    )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
