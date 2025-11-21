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


def get_ground_truth(slide_path: Path) -> int:
    ground_truth: str | int = slide_path.stem[-1]
    if ground_truth not in ("0", "1"):
        raise ValueError(
            f"Invalid slide name: {slide_path.stem}. Expected format: *-[0,1].mrxs"
        )
    return int(ground_truth)
