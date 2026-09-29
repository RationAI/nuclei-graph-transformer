"""This script generates a CSV metadataset file for the BEETLE dataset."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


@ray.remote(num_cpus=1)
def read_slide_properties(slide: dict[str, str], output_dir: str) -> None:
    """Saves level-0 extent and mpp of the slide, or an error message if it can't be opened."""
    try:
        with OpenSlide(slide["slide_path"]) as slide_file:
            extent_x, extent_y = slide_file.level_dimensions[0]
            mpp_x, mpp_y = slide_file.slide_resolution(0)
        result = {
            "extent_x": extent_x,
            "extent_y": extent_y,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
        }
    except KeyError:
        result = {"error": "missing MPP metadata"}
    except Exception as e:  # noqa: BLE001 - any failure means the slide is unusable
        result = {"error": f"{type(e).__name__}: {e}"}
    Path(output_dir, f"{slide['slide_id']}.json").write_text(json.dumps(result))


def get_patient_id(row: dict[str, str]) -> str | None:
    if row["patient_id"]:
        return row["patient_id"]
    if row["source"] == "tcga":
        # TCGA slides have no patient_id; use the barcode prefix
        return "-".join(row["name"].split("-")[:3])
    return None


def parse_slide_info(
    row: dict[str, str], root: Path, log_file: Path
) -> dict[str, str | bool | None] | None:
    def log(msg: str) -> None:
        with log_file.open("a") as f:
            f.write(msg + "\n")

    if not row.get("name"):
        return None

    slide_path = root / "data" / row["wsi_path"]
    if not row["wsi_path"] or not slide_path.is_file():
        log(f"SLIDE_MISSING: {row['name']} ({slide_path})")
        return None

    mask_path = (
        root / row["annotation_mask_path"] if row["annotation_mask_path"] else None
    )
    xml_path = root / row["annotation_xml_path"] if row["annotation_xml_path"] else None
    json_path = (
        root / row["annotation_json_path"] if row["annotation_json_path"] else None
    )

    return {
        "slide_id": row["name"],
        "slide_path": str(slide_path),
        "mask_path": str(mask_path) if mask_path and mask_path.exists() else "",
        "has_annotation": bool(mask_path and mask_path.exists()),
        "has_annotation_xml": bool(xml_path and xml_path.exists()),
        "has_annotation_json": bool(json_path and json_path.exists()),
        "patient_id": get_patient_id(row),
        "source": row["source"],
        "specimen_type": row["specimen_type"],
        "scanner": row["scanner"],
        "split": row["split"],
        "validation_fold": row["validation_fold"] or None,
    }


def get_dataframes(
    overview_csv: Path, root: Path, max_concurrent: int, log_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with overview_csv.open(newline="") as f:
        records = [parse_slide_info(row, root, log_file) for row in csv.DictReader(f)]
    df = pd.DataFrame([r for r in records if r is not None])

    with TemporaryDirectory() as properties_dir:
        process_items(
            df[["slide_id", "slide_path"]].to_dict("records"),
            process_item=read_slide_properties,
            fn_kwargs={"output_dir": properties_dir},
            max_concurrent=max_concurrent,
        )
        results = {
            p.name.removesuffix(".json"): json.loads(p.read_text())
            for p in Path(properties_dir).glob("*.json")
        }

    properties = {sid: r for sid, r in results.items() if "error" not in r}
    with log_file.open("a") as f:
        f.writelines(
            f"SLIDE_UNREADABLE: {sid} - {r['error']}\n"
            for sid, r in results.items()
            if "error" in r
        )

    df = df[df["slide_id"].isin(properties)].reset_index(drop=True)
    df = df.join(pd.DataFrame.from_dict(properties, orient="index"), on="slide_id")

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
    ray.init(num_cpus=config.max_concurrent)

    with TemporaryDirectory() as output_dir:
        df, summary_df = get_dataframes(
            overview_csv=Path(config.overview_csv),
            root=Path(config.root),
            max_concurrent=config.max_concurrent,
            log_file=Path(output_dir) / "errors.log",
        )

        df.to_csv(Path(output_dir) / "slides_metadata.csv", index=False)
        summary_df.to_csv(Path(output_dir) / "summary.csv", index=False)

        logger.log_artifacts(local_dir=output_dir, artifact_path="beetle")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df, name="beetle")
        mlflow.log_input(slide_dataset, context="slides_metadata")

    ray.shutdown()


if __name__ == "__main__":
    main()
