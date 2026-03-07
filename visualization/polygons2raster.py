"""Script for creating a visualization of the nuclei segmentation results and their labels.

Assumes the following structure of input data:
1. Segmented nuclei (`preprocessing/nuclei_segmentation.py`):
<NUCLEI_PATH>/
    <DATASET_NAME>/
        slide_id=<SLIDE_NAME>/
            *.parquet (columns "id" (str), "polygon" (np.ndarray[float]) and "centroid" (np.ndarray[float]))

2. (Optional) Model Predictions (`nuclei_graph/callbacks/prediction_labels.py`):
<PREDICTIONS_URI>/
    <SLIDE_NAME>.parquet (columns "id" (str), "prediction" (int))

3. (Optional) Heatmap labels (`preprocessing/unipolar_heatmap_labels.py`) for positive slides:
<HEATMAP_LABELS_URI>/
    <SLIDE_NAME>.parquet (columns "slide_id" (str), "id" (str), and <LABEL_COLUMN> (int))

4. (Optional) CAM labels (`preprocessing/cam_labels.py`):
<CAM_LABELS_URI>/
    <SLIDE_NAME>.parquet (columns "slide_id" (str), "id" (str), "cam_label" (int), and "cam_score" (float))

Visualization Modes:
1) Outline: Only outline the segmented nuclei polygons.
2) Predictions: Creates nuclei masks according to model predictions — nuclei predicted as positive are filled;
    `predictions_uri` and `pred_thr` must be provided.
3) Heatmap-based Labeling: Creates nuclei masks for positive slides according to heatmap labels — nuclei
    inside heatmaps are filled; `heatmap_labels_uri` must be provided.
4) CAM-based Pseudo Labeling: Creates nuclei masks for positive slides according to CAM pseudo-labels — nuclei
    inside specified high-confidence CAM regions (positive or negative) are filled; `cam_labels_uri` must be provided.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import pyvips
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from rationai.masks import process_items, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


def set_filling_and_get_outline_color(
    nuclei: pd.DataFrame,
    visualization_mode: int,
    slide_path: Path,
    heatmap_labels_dir: Path | None,
    label_column: str | None,
    cam_labels_dir: Path | None,
    predictions_dir: Path | None,
    pred_thr: float | None,
) -> tuple[pd.DataFrame, int | None]:
    nuclei["fill_color"] = None
    outline_color = None
    match visualization_mode:
        case 1:  # Outline
            outline_color = 255

        case 2:  # Predictions
            assert predictions_dir is not None and pred_thr is not None
            predictions_path = predictions_dir / f"{slide_path.stem}.parquet"
            if not predictions_path.exists():  # slides from the train set
                return nuclei, outline_color
            predictions_df = pd.read_parquet(predictions_path)
            nuclei = nuclei.merge(predictions_df, on="id", how="inner")
            nuclei.loc[nuclei["prediction"] >= pred_thr, "fill_color"] = 255

        # --- Modes used for a visual check of the preprocessing steps ---
        case 3:  # Heatmap-based Labeling
            assert heatmap_labels_dir is not None and label_column is not None
            heatmap_path = heatmap_labels_dir / f"{slide_path.stem}.parquet"
            if not heatmap_path.exists():  # negative slide
                return nuclei, outline_color
            heatmap_df = pd.read_parquet(heatmap_path)
            nuclei = nuclei.merge(heatmap_df, on="id", how="inner")
            nuclei.loc[nuclei[label_column] == 1, "fill_color"] = 255

        case 4:  # CAM-based Pseudo Labeling
            assert cam_labels_dir is not None
            cam_path = cam_labels_dir / f"{slide_path.stem}.parquet"
            if not cam_path.exists():  # negative slide
                return nuclei, outline_color
            cam_df = pd.read_parquet(cam_path)
            nuclei = nuclei.merge(cam_df, on="id", how="inner")
            # fill both positive and negative regions (or pick one class to fill; comment out as needed)
            nuclei.loc[nuclei["cam_label"] == 1, "fill_color"] = 255
            nuclei.loc[nuclei["cam_label"] == 0, "fill_color"] = 255

    return nuclei, outline_color


@ray.remote(memory=90 * 1024**3)
def process_slide(
    slide_path: Path,
    visualization_mode: int,
    mpp: float,
    mask_tile_width: int,
    mask_tile_height: int,
    nuclei_dir: Path,
    output_dir: Path,
    label_dirs: dict[str, Path | None],
    pred_thr: float | None,
) -> None:
    dataset_name = slide_path.parents[0].name
    nuclei = pd.read_parquet(nuclei_dir / dataset_name / f"slide_id={slide_path.stem}")

    nuclei, outline_color = set_filling_and_get_outline_color(
        nuclei, visualization_mode, slide_path, **label_dirs, pred_thr=pred_thr
    )

    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(mpp)
        mask_mpp_x, mask_mpp_y = slide.slide_resolution(level)
        mask_size = slide.level_dimensions[level]
    mask = Image.new("L", size=mask_size)
    canvas = ImageDraw.Draw(mask)

    for row in nuclei.itertuples(index=False):
        if row.fill_color is not None:
            canvas.polygon(xy=row.polygon, outline=None, fill=row.fill_color)
        if outline_color is not None:  # use line with width param for better visibility
            poly = row.polygon.tolist()
            closed_poly = poly + poly[:2]
            canvas.line(xy=closed_poly, fill=outline_color, width=7)

    write_big_tiff(
        image=pyvips.Image.new_from_array(mask),
        path=output_dir / slide_path.with_suffix(".tiff").name,
        mpp_x=mask_mpp_x,
        mpp_y=mask_mpp_y,
        tile_width=mask_tile_width,
        tile_height=mask_tile_height,
    )


def get_local_path(uri: str | None) -> Path | None:
    return Path(download_artifacts(uri)) if uri is not None else None


@with_cli_args(["+visualization=polygons2raster"])
@hydra.main(config_path="../configs", config_name="visualization", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert config.visualization_mode in {1, 2, 3, 4}

    train_slides = pd.read_csv(download_artifacts(config.train_metadata_uri))
    test_slides = pd.read_csv(download_artifacts(config.test_metadata_uri))
    slides = pd.concat([train_slides, test_slides])
    valid_slides = slides[~slides["is_carcinoma"] | slides["has_annotation"]]

    label_dirs = {
        "heatmap_labels_dir": get_local_path(config.heatmap_labels_uri),
        "cam_labels_dir": get_local_path(config.cam_labels_uri),
        "predictions_dir": get_local_path(config.predictions_uri),
    }

    with TemporaryDirectory() as output_dir:
        process_items(
            items=[
                Path(
                    "/mnt/data/MOU/prostate/tile_level_annotations_test/TP-2019_2623-12-1.mrxs"
                )
            ],  # valid_slides["slide_path"].map(Path),
            process_item=process_slide,
            fn_kwargs={
                "visualization_mode": int(config.visualization_mode),
                "mpp": config.mpp,
                "mask_tile_width": config.mask_tile_width,
                "mask_tile_height": config.mask_tile_height,
                "nuclei_dir": Path(config.nuclei_path),
                "output_dir": Path(output_dir),
                "label_dirs": label_dirs,
                "pred_thr": config.get("pred_thr", None),
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(
            local_dir=output_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
