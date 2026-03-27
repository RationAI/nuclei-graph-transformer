"""Script for mapping slide-level metadata for the PANDA dataset.

The mapping is intended to be used within the DataModule class for downstream data loading.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
import pyarrow.parquet as pq
from mlflow.artifacts import download_artifacts
from mlflow.data import pandas_dataset
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@with_cli_args(["+preprocessing=metadata_mapping"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = pd.read_csv(Path(download_artifacts(config.metadata_uri)))
    valid_slides = slides[
        slides["is_carcinoma"]
        & slides["has_annotation"]
        & slides["has_segmentation"]
        & ~slides["is_annotation_corrupted"]
        & slides["is_wsi_valid"]
    ]

    nuclei_dir = Path(config.nuclei_path)
    nuclei_paths = valid_slides["slide_id"].map(
        lambda id: nuclei_dir / f"slide_id={id}"
    )
    nuclei_counts = nuclei_paths.map(str).apply(
        lambda path: sum(f.metadata.num_rows for f in pq.ParquetDataset(path).fragments)
    )

    map_df = pd.DataFrame(
        {
            "slide_id": valid_slides["slide_id"],
            "slide_path": valid_slides["slide_path"],
            "slide_nuclei_path": nuclei_paths.map(str),
            "nuclei_count": nuclei_counts.astype("Int64"),
            "is_carcinoma": valid_slides["is_carcinoma"],
            "data_provider": valid_slides["data_provider"],
            "mpp_x": valid_slides["mpp_x"],
            "mpp_y": valid_slides["mpp_y"],
        }
    )
    with TemporaryDirectory() as output_dir:
        parquet_path = Path(output_dir, "slides_mapping.parquet")
        map_df.to_parquet(parquet_path, index=False)
        logger.log_artifact(local_path=str(parquet_path), artifact_path="panda")
        slide_dataset = pandas_dataset.from_pandas(map_df, name="panda_map")
        mlflow.log_input(slide_dataset, context="slides_mapping")


if __name__ == "__main__":
    main()
