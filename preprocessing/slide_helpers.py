"""Small helpers for reading slide metadata often used in preprocessing scripts.

The following functions are intended to extract metadata from CSV file (generated
by `exploration/save_metadataset.py`) located in the provided path `metadata_path`.

The CSV is expected to contain the following columns:
    - `slide_path` (str): Slide storage path (.mrxs).
    - `has_annotation` (bool): True if an XML annotation file exists for the slide.
    - `is_carcinoma` (bool): True if the slide is positive (`*-1.mrxs`).
    - `patient_id` (str): 4-digit unique patient ID (for patient-level train/validation set splitting)
"""

from pathlib import Path

import pandas as pd


def get_ground_truth(slide_path: Path) -> int:
    ground_truth: str | int = slide_path.stem[-1]
    if ground_truth not in ("0", "1"):
        raise ValueError(
            f"Invalid slide name: {slide_path.stem}. Expected format: *-[0,1].mrxs"
        )
    return int(ground_truth)


def get_positive_slides(metadata_path: Path) -> list[Path]:
    df = pd.read_csv(metadata_path)
    df_with_annot = df[df["is_carcinoma"]]
    return df_with_annot["slide_path"].map(Path).tolist()


def get_positive_slides_with_annots(metadata_path: Path) -> list[Path]:
    df = pd.read_csv(metadata_path)
    df_with_annot = df[df["has_annotation"] & (df["is_carcinoma"])]
    return df_with_annot["slide_path"].map(Path).tolist()


def get_missing_annotations(metadata_path: Path) -> list[Path]:
    df = pd.read_csv(metadata_path)
    df_missing_annot = df[~df["has_annotation"] & (df["is_carcinoma"])]
    return df_missing_annot["slide_path"].map(Path).tolist()


def get_cleaned_slides(
    metadata_path: Path,
    missing_annotation_slides_path: Path,
    missing_cam_slides_path: Path,
) -> pd.DataFrame:
    """Filters slides in the provided CSV `metadata_path` excluding slides with missing annotations or CAM masks.

    Assumes `missing_annotation_slides` and `missing_cam_slides` CSV files have a column
    named `slide_path` containing the paths to the slides to be excluded.
    """
    df = pd.read_csv(metadata_path)
    missing_annotation_slides = pd.read_csv(missing_annotation_slides_path)
    missing_cam_slides = pd.read_csv(missing_cam_slides_path)

    exclude_slides = pd.concat(
        [missing_annotation_slides["slide_path"], missing_cam_slides["slide_path"]],
        ignore_index=True,
    ).drop_duplicates()

    cleaned_df = df[~df["slide_path"].isin(exclude_slides)]
    return cleaned_df
