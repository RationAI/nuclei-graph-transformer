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


def log_input(df: pd.DataFrame, name: str, logger: MLFlowLogger) -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        logger.log_artifact(str(path), artifact_path="panda")
        mlflow.log_input(
            pandas_dataset.from_pandas(df, name=name),
            context=f"panda/{name}",
        )


@with_cli_args(["+preprocessing=metadata_mapping/panda"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = pd.read_csv(download_artifacts(config.metadata_uri))
    split = pd.read_csv(download_artifacts(config.split_uri))
    slides = slides.merge(split, on="slide_id", how="inner")

    nuclei_dir = Path(config.nuclei_path)
    nuclei_paths = slides["slide_id"].map(lambda id: nuclei_dir / f"slide_id={id}")
    nuclei_counts = nuclei_paths.apply(
        lambda path: pq.read_metadata(path / "nuclei.parquet").num_rows
    )

    map_df = pd.DataFrame(
        {
            "slide_id": slides["slide_id"],
            "slide_path": slides["slide_path"],
            "slide_nuclei_path": nuclei_paths.map(str),
            "nuclei_count": nuclei_counts.astype("Int64"),
            "is_carcinoma": slides["isup_grade"] >= 1,
            "mpp_x": slides["mpp_x"],
            "mpp_y": slides["mpp_y"],
            "set": slides["set"],
        }
    )

    for split_name in ["train", "test"]:
        split_df = map_df[map_df["set"] == split_name].copy()
        split_df = split_df.drop(columns=["set"])
        log_input(split_df, f"slides_mapping_{split_name}", logger)


if __name__ == "__main__":
    main()
