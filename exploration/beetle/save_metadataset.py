"""This script generates a CSV metadataset file for the BEETLE dataset.

This serves as a snapshot of the current data version used in this project further on.
"""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def parse_slide_info(
    row: dict[str, str], root: Path, log_file: Path
) -> dict[str, str | bool | None] | None:
    def log(msg: str) -> None:
        with log_file.open("a") as f:
            f.write(msg + "\n")

    if not row.get("name"):
        return None

    slide_path = root / "data" / row["wsi_path"]
    if not slide_path.exists():
        log(f"SLIDE_MISSING: {row['name']} ({slide_path})")
        return None

    mask_path = root / row["annotation_mask_path"] if row["annotation_mask_path"] else None
    xml_path = root / row["annotation_xml_path"] if row["annotation_xml_path"] else None
    json_path = root / row["annotation_json_path"] if row["annotation_json_path"] else None

    return {
        "slide_id": row["name"],
        "slide_path": str(slide_path),
        "mask_path": str(mask_path) if mask_path and mask_path.exists() else "",
        "has_annotation": bool(mask_path and mask_path.exists()),
        "has_annotation_xml": bool(xml_path and xml_path.exists()),
        "has_annotation_json": bool(json_path and json_path.exists()),
        "patient_id": row["patient_id"],
        "source": row["source"],
        "specimen_type": row["specimen_type"],
        "scanner": row["scanner"],
        "split": row["split"],
        "validation_fold": row["validation_fold"] or None,
    }


def get_dataframes(
    overview_csv: Path, root: Path, log_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with overview_csv.open(newline="") as f:
        records = [parse_slide_info(row, root, log_file) for row in csv.DictReader(f)]
    df = pd.DataFrame([r for r in records if r is not None])

    summary_df = (
        df.groupby(["source", "specimen_type", "split"])
        .agg(
            Patients=("patient_id", "nunique"),
            Total_Slides=("slide_id", "count"),
            Annotations=("has_annotation", "sum"),
        )
        .reset_index()
    )
    return df, summary_df


@with_cli_args(["+exploration=beetle/save_metadataset"])
@hydra.main(config_path="../../configs", config_name="exploration", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    with TemporaryDirectory() as output_dir:
        df, summary_df = get_dataframes(
            overview_csv=Path(config.overview_csv),
            root=Path(config.root),
            log_file=Path(output_dir) / "errors.log",
        )

        df.to_csv(Path(output_dir) / "slides_metadata.csv", index=False)
        summary_df.to_csv(Path(output_dir) / "summary.csv", index=False)

        logger.log_artifacts(local_dir=output_dir, artifact_path="beetle")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df, name="beetle")
        mlflow.log_input(slide_dataset, context="slides_metadata")


if __name__ == "__main__":
    main()
