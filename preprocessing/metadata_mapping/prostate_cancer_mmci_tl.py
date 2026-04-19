"""Script for mapping slide-level metadata.

The generated mapping is intended to be used within the DataModule class for downstream data loading.
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
from ratiopath.openslide import OpenSlide


def get_mpp(slide_path: str) -> tuple[float, float]:
    with OpenSlide(slide_path) as slide:
        return slide.slide_resolution(level=0)


def build_map(
    slides: pd.DataFrame,
    nuclei_dir: Path,
    logger: MLFlowLogger,
    dataset_name: str,
) -> None:
    slide_ids = slides["slide_path"].map(lambda path: Path(path).stem)
    nuclei_paths = slide_ids.map(
        lambda id: nuclei_dir / dataset_name / f"slide_id={id}"
    )
    nuclei_counts = nuclei_paths.map(str).apply(
        lambda path: sum(f.metadata.num_rows for f in pq.ParquetDataset(path).fragments)
    )
    mpp_x, mpp_y = zip(*slides["slide_path"].apply(get_mpp), strict=True)

    map_df = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "patient_id": slides["patient_id"].map(str),
            "slide_path": slides["slide_path"],
            "slide_nuclei_path": nuclei_paths.map(str),
            "nuclei_count": nuclei_counts.astype("Int64"),
            "is_carcinoma": slides["is_carcinoma"],
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
        }
    )
    with TemporaryDirectory() as output_dir:
        parquet_path = Path(output_dir, "slides_mapping.parquet")
        map_df.to_parquet(parquet_path, index=False)

        logger.log_artifact(
            local_path=str(parquet_path),
            artifact_path=f"{dataset_name}",
        )
        slide_dataset = pandas_dataset.from_pandas(map_df, name=f"{dataset_name}_map")
        mlflow.log_input(slide_dataset, context="slides_mapping")


@with_cli_args(["+preprocessing=metadata_mapping/prostate_cancer_mmci_tl"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    exclusion_batches: list[pd.Series] = []
    for uri in config.exclude_slides_uris:
        df = pd.read_csv(download_artifacts(uri))
        exclusion_batches.append(df["slide_path"])

    exclude_slides = (
        pd.concat(exclusion_batches, ignore_index=True).drop_duplicates()
        if exclusion_batches
        else pd.Series(dtype=str)
    )

    train_slides = pd.read_csv(download_artifacts(config.train_metadata_uri))
    build_map(
        slides=train_slides[~train_slides["slide_path"].isin(exclude_slides)],
        nuclei_dir=Path(config.nuclei_seg_path),
        logger=logger,
        dataset_name=Path(config.train_data_path).name,
    )

    test_slides = pd.read_csv(download_artifacts(config.test_metadata_uri))
    build_map(
        slides=test_slides[~test_slides["slide_path"].isin(exclude_slides)],
        nuclei_dir=Path(config.nuclei_seg_path),
        logger=logger,
        dataset_name=Path(config.test_data_path).name,
    )


if __name__ == "__main__":
    main()
