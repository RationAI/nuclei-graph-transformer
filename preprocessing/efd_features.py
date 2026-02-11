"""Script for converting nuclei polygons to Elliptic Fourier Descriptor (EFD) features and extracting their scales and orientation angles.

Assumes the following structure of input data:
1. Metadata Mapping (`preprocessing/metadata_mapping.py`):
<DATASET_NAME>/
    slides_mapping.parquet (columns "slide_id" (str), "slide_nuclei_path" (str)).

The result is logged to MLflow as:
<MLFLOW_ARTIFACT_PATH>/
    <SLIDE_NAME>.parquet (columns "slide_id" (str), "id" (str), and "efds" (np.ndarray[float]), "efd_scales" (np.ndarray[float]), "efd_angles" (np.ndarray[float]))
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

    efd = elliptic_fourier_descriptors(np.asarray(contours), efd_order)
    efd, angles = normalize_efd_for_rotation(efd)
    efd, scales = normalize_efd_for_scale(efd)
    efd = rearrange(efd, "n order c -> n (order c)")
    # after scale normalization A1, B1, and C1 coeffs are fixed constants so we can remove them
    efd = efd[:, 3:]

    efd_df = pd.DataFrame(
        {
            "slide_id": slide_id,
            "id": nuclei["id"],
            "efd": list(efd),
            "efd_scale": scales.flatten(),
            "efd_angle": angles.flatten(),
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
    slides = pd.concat([train_slides, test_slides])
    slides = slides[["slide_id", "slide_nuclei_path"]]

    with TemporaryDirectory() as tmp_dir:
        items = [
            (sid, Path(path))
            for sid, path in zip(
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
        logger.log_artifacts(
            local_dir=tmp_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
