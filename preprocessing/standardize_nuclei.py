"""Script for standardizing provided nuclei segmentation data.

This script processes given nuclei segmentation files, converts radial distances
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


@ray.remote(num_cpus=1, memory=(1 * 1024**3))
def standardize_nuclei(
    item: dict,
    output_dir: Path,
) -> None:
    partition = item["nuclei_partition"]
    slide_id = item["slide_id"]

    nuclei = pd.read_parquet(partition, columns=["points", "radial_distances"])
    nuclei["id"] = [
        hashlib.sha256(f"{slide_id}_{partition.stem}_{i}".encode()).hexdigest()
        for i in nuclei.index
    ]

    # convert radial distances to Cartesian coordinates
    points = np.stack(nuclei["points"].tolist())
    radial_distances = np.stack(nuclei["radial_distances"].tolist())

    num_vertices = radial_distances.shape[-1]
    t = np.linspace(0, 1, num_vertices + 1, dtype=np.float32)[:-1]
    cos = np.cos(2 * np.pi * t)
    sin = np.sin(2 * np.pi * t)

    polar = radial_distances[..., None] * np.stack([sin, cos], axis=-1)
    polygons = points[:, None, :] + polar

    nuclei["polygon"] = [poly.flatten() for poly in polygons]
    nuclei["centroid"] = [c.astype(np.float32) for c in polygons.mean(axis=1)]

    output_path = output_dir / f"slide_id={slide_id}" / partition.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nuclei[["id", "polygon", "centroid"]].to_parquet(output_path, index=False)


@with_cli_args(["+preprocessing=standardize_nuclei"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, _: MLFlowLogger) -> None:
    nuclei_dir = Path(config.nuclei_source_path)
    metadata = pd.read_csv(Path(download_artifacts(config.metadata_uri)))

    items_to_process = []

    for row in metadata.itertuples(index=False):
        slide_dir = nuclei_dir / f"slide_id={row.id}"
        for parquet_file in slide_dir.glob("*.parquet"):
            items_to_process.append(
                {
                    "nuclei_partition": parquet_file,
                    "slide_id": str(row.slide_id),
                }
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
