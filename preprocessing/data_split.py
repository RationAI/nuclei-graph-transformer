import tempfile
from pathlib import Path

import hydra
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split


@with_cli_args(["+preprocessing=data_split"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = pd.read_csv(Path(download_artifacts(config.metadata_uri)))
    slides = slides[
        slides["has_annotation"]
        & slides["has_segmentation"]
        & ~slides["is_annotation_corrupted"]
        & slides["is_wsi_valid"]
    ]

    if config.restriction is not None:
        slides = slides[
            slides[config.restriction.subset_column] == config.restriction.target_value
        ]

    train, test = train_test_split(
        slides,
        test_size=config.test_size,
        random_state=42,
        stratify=slides[config.stratify_column],
    )
    train["set"] = "train"
    test["set"] = "test"
    split = pd.concat([train, test], ignore_index=True)

    summary = (
        split.groupby(["set", config.stratify_column]).size().reset_index(name="count")
    )
    summary[f"{config.stratify_column}_ratio"] = summary.groupby("set")[
        "count"
    ].transform(lambda x: round(x / x.sum(), 3))

    with tempfile.TemporaryDirectory() as output_dir:
        split[["slide_id", "set"]].to_csv(Path(output_dir) / "split.csv", index=False)
        logger.log_artifact(str(Path(output_dir) / "split.csv"))
        summary.to_csv(Path(output_dir) / "summary.csv", index=False)
        logger.log_artifact(str(Path(output_dir) / "summary.csv"))


if __name__ == "__main__":
    main()
