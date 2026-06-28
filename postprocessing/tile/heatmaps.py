"""Builds per-slide heatmaps of tile-level predictions.

For each slide, loads its pooled tile predictions and rasterizes them into a scalar
heatmap using. The resulting heatmaps are logged as artifacts on the MLflow run.
"""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from postprocessing.mlflow_utils import setup_mlflow
from postprocessing.tile.data_loading import get_predictions


def build_slide_heatmap(
    slide_id: str,
    slide_preds: pd.DataFrame,
    slide_row: pd.Series,
    metadata_row: pd.Series,
    level: int,
    save_dir: Path,
) -> Path:
    mpp_x = slide_row["mpp_x"] if "mpp_x" in slide_row else metadata_row["mpp_x"]
    mpp_y = slide_row["mpp_y"] if "mpp_y" in slide_row else metadata_row["mpp_y"]

    with OpenSlide(Path(metadata_row["slide_path"])) as slide:
        downsample = slide.level_downsamples[level]

    extent_tile = slide_row["tile_extent_x"]
    stride = slide_row.get("stride_x", extent_tile)

    mask_builder = ScalarMaskBuilder(
        save_dir=save_dir,
        filename=str(slide_id),
        extent_x=round(int(slide_row["extent_x"]) / downsample),
        extent_y=round(int(slide_row["extent_y"]) / downsample),
        mpp_x=float(mpp_x) * downsample,
        mpp_y=float(mpp_y) * downsample,
        extent_tile=round(int(extent_tile) / downsample),
        stride=round(int(stride) / downsample),
        device="cpu",
    )

    preds = torch.as_tensor(
        slide_preds["tile_prediction"].to_numpy(), dtype=torch.float32
    ).unsqueeze(-1)
    xs = torch.as_tensor(slide_preds["x"].to_numpy() / downsample, dtype=torch.float32)
    ys = torch.as_tensor(slide_preds["y"].to_numpy() / downsample, dtype=torch.float32)

    mask_builder.update(preds, xs, ys)
    return mask_builder.save()


@with_cli_args(["+postprocessing=tile/heatmaps"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert logger.run_id is not None
    tag_parent_run(logger.run_id, config.get("mlflow_parent_run_id"))

    predictions_dir = Path(download_artifacts(config.pooled_predictions_uri))
    preds_df = get_predictions(predictions_dir)

    slides_df = pd.read_parquet(download_artifacts(config.slides_uri)).set_index("stem")
    metadata_df = pd.read_parquet(download_artifacts(config.metadata_uri)).set_index(
        "slide_id"
    )

    client, mlflow_run_id = setup_mlflow(config)
    assert mlflow_run_id is not None

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for slide_id, slide_preds in preds_df.groupby("slide_id"):
            build_slide_heatmap(
                slide_id,
                slide_preds,
                slides_df.loc[slide_id],
                metadata_df.loc[slide_id],
                config.level,
                tmp_path,
            )
            gc.collect()

        client.log_artifacts(
            run_id=mlflow_run_id,
            local_dir=str(tmp_path),
            artifact_path=config.mlflow_artifact_path,
        )


if __name__ == "__main__":
    main()
