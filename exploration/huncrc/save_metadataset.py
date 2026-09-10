from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


def get_existing_annotation_ids(patch_dirs: list[str]) -> set[str]:
    """Scans directories and returns a set of slide IDs for which annotations exist."""
    ids = set()
    for p_dir in patch_dirs:
        if not Path(p_dir).exists():
            continue
        for file_path in Path(p_dir).rglob("*_labels.csv"):
            slide_id = file_path.name.replace("_labels.csv", "")
            ids.add(slide_id)
    return ids


def get_wsi_metadata(slide_path: str) -> pd.Series:
    with OpenSlide(slide_path) as slide:
        extent_x, extent_y = slide.level_dimensions[0]
        mpp_x, mpp_y = slide.slide_resolution(0)
    return pd.Series([extent_x, extent_y, mpp_x, mpp_y])


def load_base_metadata(csv_path: Path, exclude_slides: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path).rename(
        columns={
            "slideID": "slide_id",
            "CATEGORY": "category",
            "Polyp type": "polyp_type",
            "Biopsy or polyp": "sample_origin",
            "Haggit-level": "haggit_level",
        }
    )
    # Format to 3-digit string to match .mrxs filenames
    df["slide_id"] = df["slide_id"].astype(str).str.zfill(3)

    if exclude_slides:
        df = df[~df["slide_id"].isin(exclude_slides)]
    return df


def get_segmentation_props(df: pd.DataFrame, pq_path: Path) -> pd.DataFrame:
    props_df = pd.read_parquet(pq_path)
    props_df["slide_id"] = [Path(p).stem for p in props_df["path"]]
    props_df = props_df.rename(
        columns={
            "id": "segmentation_id",
            "extent_x": "seg_extent_x",
            "extent_y": "seg_extent_y",
            "mpp_x": "seg_mpp_x",
            "mpp_y": "seg_mpp_y",
        }
    )
    return df.merge(
        props_df[
            [
                "slide_id",
                "segmentation_id",
                "seg_extent_x",
                "seg_extent_y",
                "seg_mpp_x",
                "seg_mpp_y",
            ]
        ],
        on="slide_id",
        how="left",
    )


def get_dataframes(
    metadata_csv_path: Path,
    slides_dir: Path,
    patch_dirs_z1: list[str],
    patch_dirs_z2: list[str],
    properties_pq_path: Path,
    exclude_slides: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_base_metadata(metadata_csv_path, exclude_slides)
    df = get_segmentation_props(df, properties_pq_path)

    # Extract WSI Metadata (Level 0)
    df["slide_path"] = df["slide_id"].apply(lambda sid: str(slides_dir / f"{sid}.mrxs"))
    df[["extent_x", "extent_y", "mpp_x", "mpp_y"]] = df["slide_path"].apply(
        get_wsi_metadata
    )

    # Check Annotations & Nuclei Segmentations
    df["has_zoom_1_annotations"] = df["slide_id"].isin(
        get_existing_annotation_ids(patch_dirs_z1)
    )
    df["has_zoom_2_annotations"] = df["slide_id"].isin(
        get_existing_annotation_ids(patch_dirs_z2)
    )
    df["has_segmentation"] = df["segmentation_id"].notna()

    # Aggregation Statistics
    summary_df = (
        df.groupby(
            ["haggit_level", "sample_origin", "polyp_type", "category"], dropna=False
        )
        .agg(
            Total_Slides=("slide_id", "count"),
            Z1_Annotations=("has_zoom_1_annotations", "sum"),
            Z2_Annotations=("has_zoom_2_annotations", "sum"),
        )
        .reset_index()
    )

    final_cols = [
        "slide_id",  # 3-digit number between 1 and 200 (anonymous patient ID)
        "slide_path",
        "segmentation_id",  # ID of the slide in the parquet dataset with segmented nuclei
        "has_segmentation",  # True if the nuclei segmentation file exists
        "has_zoom_1_annotations",  # True if zoom 1 patch annotations exist
        "has_zoom_2_annotations",  # True if zoom 2 patch annotations exist
        # Global Annotations
        "category",
        "polyp_type",
        "sample_origin",
        "haggit_level",
        # Level 0 WSI Properties
        "extent_x",
        "extent_y",
        "mpp_x",
        "mpp_y",
        # Nuclei Segmentation Properties
        "seg_extent_x",
        "seg_extent_y",
        "seg_mpp_x",
        "seg_mpp_y",
    ]
    return df[final_cols], summary_df


def uris2df(uris: list[str]) -> pd.DataFrame:
    """Loads and merges multiple metadata .CSV files into a single DataFrame."""
    batches = [pd.read_csv(download_artifacts(uri)) for uri in uris]
    return pd.concat(batches, ignore_index=True)


@with_cli_args(["+exploration=huncrc/save_metadataset"])
@hydra.main(config_path="../../configs", config_name="exploration", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    exclude_slides = []
    if config.exclude_slides:
        exclude_slides = uris2df(config.exclude_slides)["slide_stem"].tolist()

    with TemporaryDirectory() as output_dir:
        df, summary_df = get_dataframes(
            metadata_csv_path=Path(config.metadata_csv),
            slides_dir=Path(config.slides_dir),
            patch_dirs_z1=config.patch_dirs_z1,
            patch_dirs_z2=config.patch_dirs_z2,
            properties_pq_path=Path(config.slides_properties_parquet),
            exclude_slides=exclude_slides,
        )

        df.to_csv(Path(output_dir) / "slides_metadata.csv", index=False)
        summary_df.to_csv(Path(output_dir) / "summary.csv", index=False)

        logger.log_artifacts(local_dir=output_dir, artifact_path="huncrc")
        slide_dataset = mlflow.data.pandas_dataset.from_pandas(df, name="huncrc")
        mlflow.log_input(slide_dataset, context="slides_metadata")


if __name__ == "__main__":
    main()
