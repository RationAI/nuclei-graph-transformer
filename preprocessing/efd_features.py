"""Script for converting nuclei polygons to Elliptic Fourier Descriptors (EFDs).

Assumes the following structure of input data:
1. Metadata Mapping (`preprocessing/metadata_mapping.py`):
<DATASET_NAME>/
    slides_mapping.parquet (columns "slide_id" (str), "slide_nuclei_path" (str)).

2. Segmented nuclei (`preprocessing/nuclei_segmentation.py`):
<NUCLEI_PATH>/
    <DATASET_NAME>/
        slide_id=<SLIDE_NAME>/
            *.parquet (columns "id" (str), "polygon" (np.ndarray[float]) and "centroid" (np.ndarray[float]))

The result is a PyTorch binary file for each slide containing the raw EFD features, scale factor, and orientation
angle for each nucleus. All are ordered by nucleus id.

The output is saved as:
<OUTPUT_PATH>/
    <DATASET_NAME>/
        <SLIDE_NAME>.pt (a dictionary of Tensors with keys "efds", "scales", and "angles").
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import ray
import torch
from einops import rearrange
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_rotation,
    normalize_efd_for_scale,
)


@ray.remote(memory=2 * 1024**3)
def compute_efds(data_pair: tuple[str, str], output_dir: Path, efd_order: int) -> None:
    slide_id, nuclei_path = data_pair
    nuclei = pd.read_parquet(nuclei_path, columns=["id", "polygon"]).sort_values("id")
    contours = rearrange(nuclei["polygon"].tolist(), "b (v d) -> b v d", d=2)

    efds = elliptic_fourier_descriptors(np.asarray(contours), efd_order)
    _, angles = normalize_efd_for_rotation(efds)
    _, scales = normalize_efd_for_scale(efds)

    efds = rearrange(efds, "n order c -> n (order c)")

    slide_data = {
        "efds": torch.from_numpy(efds).float(),
        "scales": torch.from_numpy(scales.flatten()).float(),
        "angles": torch.from_numpy(angles.flatten()).float(),
    }
    output_path = output_dir / f"{slide_id}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slide_data, output_path)


def load_df(uri: str, cols: list[str] | None = None) -> pd.DataFrame:
    path = download_artifacts(uri)
    return pd.read_parquet(path, columns=cols)


@with_cli_args(["+preprocessing=efd_features"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    columns = ["slide_id", "slide_nuclei_path"]
    train_slides = load_df(config.train_metadata_uri, cols=columns)
    test_slides = load_df(config.test_metadata_uri, cols=columns)

    data_sets = {
        Path(config.train_data_path).name: train_slides,
        Path(config.test_data_path).name: test_slides,
    }

    for dataset_name, slides in data_sets.items():
        items = [
            (s_id, Path(path))
            for s_id, path in zip(
                slides["slide_id"], slides["slide_nuclei_path"], strict=True
            )
        ]
        process_items(
            items=items,
            process_item=compute_efds,  # type: ignore[misc]
            fn_kwargs={
                "output_dir": Path(config.output_path) / dataset_name,
                "efd_order": config.efd_order,
            },
            max_concurrent=config.max_concurrent,
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
