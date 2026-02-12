"""Script for converting nuclei polygons to Elliptic Fourier Descriptor (EFD) features..

Assumes the following structure of input data:
1. Metadata Mapping (`preprocessing/metadata_mapping.py`):
<DATASET_NAME>/
    slides_mapping.parquet (columns "slide_id" (str), "slide_nuclei_path" (str)).

The result is a Parquet file for each slide containing the raw EFD features, rotation normalized EFD features, scale factors,
and orientation angles for each nucleus.

The output is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.parquet (columns "slide_id" (str), "id" (str), "efd_raw" (np.ndarray[float]), "efd" (np.ndarray[float]), "efd_scales" (np.ndarray[float]),
    "efd_angles" (np.ndarray[float]))
"""

from pathlib import Path
from tempfile import TemporaryDirectory

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

    efd_raw = elliptic_fourier_descriptors(np.asarray(contours), efd_order)
    efd_rotated, angles = normalize_efd_for_rotation(efd_raw)
    _, scales = normalize_efd_for_scale(efd_raw)

    efd_rotated = rearrange(efd_rotated, "n order c -> n (order c)")
    efd_raw = rearrange(efd_raw, "n order c -> n (order c)")

    efd_df = pd.DataFrame(
        {
            "slide_id": slide_id,
            "id": nuclei["id"],
            "efd_raw": list(efd_raw),
            "efd_rotated": list(efd_rotated),
            "efd_scales": scales.flatten(),
            "efd_angles": angles.flatten(),
        }
    )
    output_path = output_dir / f"{slide_id}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    efd_df.to_parquet(output_path, index=False)


@with_cli_args(["+preprocessing=efd_features"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    train_slides = pd.read_parquet(download_artifacts(config.train_metadata_uri))
    test_slides = pd.read_parquet(download_artifacts(config.test_metadata_uri))

    train_set_name = Path(config.train_data_path).name
    test_set_name = Path(config.test_data_path).name

    data_sets = {
        train_set_name: train_slides[["slide_id", "slide_nuclei_path"]],
        test_set_name: test_slides[["slide_id", "slide_nuclei_path"]],
    }

    for dataset_name, slides in data_sets.items():
        with TemporaryDirectory() as tmp_dir:
            items = [
                (id, Path(path))
                for id, path in zip(
                    slides["slide_id"], slides["slide_nuclei_path"], strict=True
                )
            ]
            process_items(
                items=items,
                process_item=compute_efds,  # type: ignore[misc]
                fn_kwargs={
                    "output_dir": Path(tmp_dir),
                    "efd_order": config.efd_order,
                },
                max_concurrent=config.max_concurrent,
            )
            logger.log_artifacts(local_dir=tmp_dir, artifact_path=dataset_name)


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
