"""Script for standardizing nuclei segmentation data.

This script processes given nuclei parquet files, converts radial distances
to Cartesian polygons and generates a globally unique ID for each nucleus.
"""

import hashlib
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(num_cpus=1, memory=(2 * 1024**3))
def standardize_nuclei(
    input_path: Path,
    output_dir: Path,
    kaggle_slide_id: str,
) -> None:
    nuclei = pd.read_parquet(input_path, columns=["points", "radial_distances"])

    nuclei["id"] = [
        hashlib.sha256(f"{kaggle_slide_id}_{i}".encode()).hexdigest()
        for i in nuclei.index
    ]

    points = np.stack(nuclei["points"].values)
    radial_distances = np.stack(nuclei["radial_distances"].values)

    num_vertices = radial_distances.shape[-1]
    t = np.linspace(0, 1, num_vertices + 1, dtype=np.float32)[:-1]
    cos = np.cos(2 * np.pi * t)
    sin = np.sin(2 * np.pi * t)

    polar = radial_distances[..., None] * np.stack([sin, cos], axis=-1)
    polygons = points[:, None, :] + polar

    nuclei["polygon"] = [poly.flatten() for poly in polygons]

    nuclei = nuclei.rename(columns={"points": "centroid"})

    output_path = output_dir / f"slide_id={kaggle_slide_id}" / input_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nuclei[["id", "polygon", "centroid"]].to_parquet(output_path, index=False)


@with_cli_args(["+preprocessing=standardize_nuclei"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, _: MLFlowLogger) -> None:
    nuclei_seg_files = Path(config.nuclei_source_path).rglob("*.parquet")
    metadata = pd.read_csv(Path(download_artifacts(config.metadata_uri)))
    id_mapping = dict(zip(metadata["id"], metadata["slide_id"], strict=True))

    items_to_process = []

    for parquet_file in nuclei_seg_files:
        internal_id = parquet_file.parent.name.split("=")[-1]

        if internal_id in id_mapping:
            items_to_process.append(
                {"nuclei_path": parquet_file, "slide_id": id_mapping[internal_id]}
            )

    process_items(
        items=items_to_process,
        process_item=standardize_nuclei,
        fn_kwargs={"output_dir": Path(config.output_path)},
        max_concurrent=config.max_concurrent,
    )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
