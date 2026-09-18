"""This script generates an exploration .CSV file for the iCAIRD Cervix dataset."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from tqdm import tqdm


@ray.remote(num_cpus=1)
def check_slide_opens(slide_path: Path, log_file: Path) -> bool:
    """Checks that OpenSlide can open the slide and read its level dimensions."""

    def log(msg: str) -> None:
        with log_file.open("a") as f:
            f.write(msg + "\n")
            f.flush()

    try:
        with OpenSlide(str(slide_path)) as slide:
            _ = slide.level_dimensions
    except Exception as e:
        log(f"SLIDE_UNREADABLE: {slide_path.stem} - {e!s}")
        return False
    return True


def parse_slide_info(
    row: dict[str, str], slides_dir: Path, annots_dir: Path, log_file: Path
) -> dict[str, str | bool] | None:
    def log(msg: str) -> None:
        with log_file.open("a") as f:
            f.write(msg + "\n")

    slide_name = row.get("slide")
    if not slide_name:
        return None

    slide_path = slides_dir / Path(slide_name).with_suffix(".tiff").name
    if not slide_path.exists():
        log(f"SLIDE_MISSING: {slide_name} ({slide_path})")
        return None

    annot_path = annots_dir / f"{slide_path.stem}.txt"

    return {
        "slide_id": slide_path.stem,
        "slide_path": str(slide_path),
        "category": row[
            "category"
        ],  # "normal_inflammation", "low_grade", "high_grade", "malignant"
        "subcategory": row.get("subcategory")
        or None,  # 'normal_inflammation', 'cin1', 'cin2', 'cgin', 'hpv', 'cin3', 'adenocarcinoma', 'squamous_carcinoma', 'other'
        "split": row.get("split") or None,
        "has_annotation": annot_path.exists(),
    }


def get_dataframes(
    metadata_csv: Path,
    slides_dir: Path,
    annots_dir: Path,
    max_concurrent: int,
    log_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with metadata_csv.open(newline="") as f:
        records = [
            parse_slide_info(row, slides_dir, annots_dir, log_file)
            for row in csv.DictReader(f)
        ]
    df = pd.DataFrame([r for r in records if r is not None])

    futures = {
        check_slide_opens.remote(Path(slide_path), log_file): slide_id
        for slide_id, slide_path in zip(df["slide_id"], df["slide_path"], strict=True)
    }
    openable = {}
    with tqdm(total=len(futures), desc="Checking slides open with OpenSlide") as pbar:
        while futures:
            done, _ = ray.wait(list(futures.keys()), num_returns=min(max_concurrent, len(futures)))
            for ref in done:
                slide_id = futures.pop(ref)
                openable[slide_id] = ray.get(ref)
            pbar.update(len(done))

    df = df[df["slide_id"].map(openable)].reset_index(drop=True)

    summary_df = (
        df.groupby(["category", "split"])
        .agg(Total_Slides=("slide_id", "count"), Annotations=("has_annotation", "sum"))
        .reset_index()
    )
    return df, summary_df


@with_cli_args(["+exploration=icaird_cervix/save_metadataset"])
@hydra.main(config_path="../../configs", config_name="exploration", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    ray.init(num_cpus=config.max_concurrent)

    with TemporaryDirectory() as output_dir:
        df, summary_df = get_dataframes(
            metadata_csv=Path(config.metadata_csv),
            slides_dir=Path(config.slides_dir),
            annots_dir=Path(config.annots_dir),
            max_concurrent=config.max_concurrent,
            log_file=Path(output_dir) / "errors.log",
        )

        df.to_csv(Path(output_dir) / "slides_metadata.csv", index=False)
        summary_df.to_csv(Path(output_dir) / "summary.csv", index=False)

        logger.log_artifacts(local_dir=output_dir, artifact_path="icaird_cervix")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df, name="icaird_cervix")
        mlflow.log_input(slide_dataset, context="slides_metadata")

    ray.shutdown()


if __name__ == "__main__":
    main()
