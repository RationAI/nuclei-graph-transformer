"""This script generates a CSV metadataset file for the `Prostate Cancer: Tile-Level Annotations Data`.

The metadata extracted for each slide include:
    - `slide_path` (str): Slide storage path (.mrxs).
    - `has_annotation` (bool): True if an XML annotation file exists for the slide.
    - `is_carcinoma` (bool): True if the slide is positive (label == 1).
    - `case_id` (str): Combination of year and patient ID (e.g., "2016_3989").
    - `patient_id` (str): 4-digit unique patient ID (for patient-level train/validation set splitting)

The resulting CSV files along with their summaries are logged as artifacts to MLflow.

This serves as a snapshot of the current data version used in this project further on.
"""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


# Expected file name format:
#     [P/TP]-[year]_[patient_id]-[slide_id]-[label].mrxs
#     where:
#         [P/TP]       - 'TP' if the slide is in the test set, 'P' otherwise
#         [year]       - Year of the slide (2016, 2019, or 2020)
#         [patient_id] - 4-digit unique patient ID
#         [slide_id]   - 2-digit slide number
#         [label]      - 0 for negative, 1 for positive
# Note: case_id is defined as [year]_[patient_id]

PATTERN = r"^(P|TP)-(?P<year>\d{4})_(?P<patient_id>\d{4})-(?P<slide_id>\d{2})-(?P<label>[01])\.mrxs$"


def parse_slide_info(slide_path: Path) -> dict[str, str | bool]:
    slide_name = slide_path.name
    match = re.match(PATTERN, slide_name, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Invalid filename format: {slide_name}\n"
            f"Expected format is [P/TP]-[year]_[patient_id]-[slide_id]-[label].mrxs"
        )

    annot_path = Path(slide_path).with_suffix(".xml")
    metadata = match.groupdict()
    return {
        "slide_path": str(slide_path),
        "has_annotation": annot_path.exists(),
        "is_carcinoma": bool(int(metadata["label"])),
        "case_id": f"{metadata['year']}_{metadata['patient_id']}",
        "patient_id": metadata["patient_id"],
    }


def get_df_summary(df: pd.DataFrame) -> pd.DataFrame:
    # case is considered positive if any slide in the case is positive.
    df["case_is_positive"] = df.groupby("case_id")["is_carcinoma"].transform("max")

    summary = {
        "Positive": {
            "Cases": df[df["case_is_positive"]]["case_id"].nunique(),
            "Patients": df[df["case_is_positive"]]["patient_id"].nunique(),
            "Slides": len(df[df["is_carcinoma"]]),
            "Slide Annotations": int(df[df["is_carcinoma"]]["has_annotation"].sum()),
        },
        "Negative": {
            "Cases": df[~df["case_is_positive"]]["case_id"].nunique(),
            "Patients": df[~df["case_is_positive"]]["patient_id"].nunique(),
            "Slides": len(df[~df["is_carcinoma"]]),
            "Slide Annotations": int(df[~df["is_carcinoma"]]["has_annotation"].sum()),
        },
        "Total": {
            "Cases": df["case_id"].nunique(),
            "Patients": df["patient_id"].nunique(),
            "Slides": len(df),
            "Slide Annotations": int(df["has_annotation"].sum()),
        },
    }
    df.drop(columns=["case_is_positive"], inplace=True)

    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    summary_df.index.name = "Category"
    return summary_df


@with_cli_args(["+exploration=save_metadataset"])
@hydra.main(config_path="../configs", config_name="exploration", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    train_data_paths = list(Path(config.train_data_path).glob("*.mrxs"))
    test_data_paths = list(Path(config.test_data_path).glob("*.mrxs"))

    data_sets = {
        Path(config.train_data_path).name: train_data_paths,
        Path(config.test_data_path).name: test_data_paths,
    }

    for dataset_name, paths in data_sets.items():
        with TemporaryDirectory() as output_dir:
            records = [parse_slide_info(path) for path in paths]
            df = pd.DataFrame(records)
            df.to_csv(Path(output_dir, "slides_metadata.csv"), index=False)

            summary_df = get_df_summary(df)
            summary_df.to_csv(Path(output_dir, "summary.csv"), index=True)

            logger.log_artifacts(local_dir=output_dir, artifact_path=dataset_name)
            slide_dataset = mlflow.data.pandas_dataset.from_pandas(
                df, name=dataset_name
            )
            mlflow.log_input(slide_dataset, context="slides_metadata")


if __name__ == "__main__":
    main()
