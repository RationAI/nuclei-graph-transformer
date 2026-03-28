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
    slides = slides[slides["has_annotation"] & slides["has_segmentation"]]

    if config.restriction is not None:
        provider_col = config.restriction.provider_column
        provider_val = config.restriction.provider_value
        slides = slides[slides[provider_col] == provider_val]

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
    summary[f"{config.stratify_column}_ratio"] = (
        summary.groupby("set")["count"].transform(lambda x: x / x.sum()).round(3)
    )
    total_counts = split["set"].value_counts().reset_index()
    total_counts.columns = ["set", "total"]

    with tempfile.TemporaryDirectory() as output_dir:
        split[["slide_id", "set"]].to_csv(Path(output_dir) / "split.csv", index=False)
        summary.to_csv(Path(output_dir) / "summary.csv", index=False)
        total_counts.to_csv(Path(output_dir) / "total_counts.csv", index=False)
        logger.log_artifacts(local_dir=output_dir, artifact_path="panda")


if __name__ == "__main__":
    main()
