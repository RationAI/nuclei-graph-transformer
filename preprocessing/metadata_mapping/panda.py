"""Script for mapping slide-level metadata for the PANDA dataset.

The mapping is intended to be used within the DataModule class for downstream data loading.
"""

from pathlib import Path
from shlex import split
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
        slides["has_annotation"]
        & slides["has_segmentation"]
        & ~slides["is_annotation_corrupted"]
        & slides["is_wsi_valid"]
    ]
    
    split = pd.read_csv(Path(download_artifacts(config.split_uri)))
    valid_slides = slides.merge(split, on="slide_id", how="inner")

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
            "is_carcinoma": valid_slides["isup_grade"] >= 1,
            "data_provider": valid_slides["data_provider"],
            "mpp_x": valid_slides["mpp_x"],
            "mpp_y": valid_slides["mpp_y"],
            "set": valid_slides["set"],
        }
    )
    
    train_df = map_df[map_df["set"] == "train"]
    test_df = map_df[map_df["set"] == "test"]
    
    with TemporaryDirectory() as output_dir:
        train_path = Path(output_dir, "slides_mapping_train.parquet")
        test_path = Path(output_dir, "slides_mapping_test.parquet")

        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

        logger.log_artifact(str(train_path), artifact_path="panda")
        logger.log_artifact(str(test_path), artifact_path="panda")

        mlflow.log_input(
            pandas_dataset.from_pandas(train_df, name="panda_map_train"),
            context="slides_mapping_train",
        )
        mlflow.log_input(
            pandas_dataset.from_pandas(test_df, name="panda_map_test"),
            context="slides_mapping_test",
        )


if __name__ == "__main__":
    main()
