"""Pools per-nucleus predictions into per-tile predictions for every slide.

For each slide, reads its nuclei-level predictions and the dataset's tiles,
pools the nuclei predictions within each tile's inner ROI into a single tile
score (via `pool_slide_tiles` from `pooling_utils.py`), and writes one parquet
file per slide containing the "x", "y", and "tile_prediction" columns.
All per-slide parquet files are then logged as artifacts on the MLflow run.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
from hydra.utils import instantiate
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.data.datamodules.base import METADATA_COLS_EVAL
from nuclei_graph.mlflow_utils import tag_parent_run
from postprocessing.mlflow_utils import setup_mlflow
from postprocessing.tile.pooling_utils import pool_slide_tiles


@with_cli_args(["+postprocessing=tile/nuclei_to_tile_pooling"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert logger.run_id is not None
    tag_parent_run(logger.run_id, config.get("mlflow_parent_run_id"))

    nuclei_predictions_dir = Path(download_artifacts(config.nuclei_predictions_uri))

    slides_df = pd.read_parquet(
        download_artifacts(config.metadata_uri), columns=METADATA_COLS_EVAL
    )
    dataset = instantiate(config.data.dataset, metadata=slides_df)

    client, mlflow_run_id = setup_mlflow(config)
    assert mlflow_run_id is not None

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for stem, tiles in dataset.tiles.groupby("stem"):
            nuclei_preds = pd.read_parquet(
                nuclei_predictions_dir / f"{stem}.parquet"
            ).set_index("id")["nuclei_prediction"]

            tile_preds_df = pool_slide_tiles(
                tiles, dataset, nuclei_preds, config.pooling_mode, config.pooling_k
            )
            tile_preds_df.to_parquet(
                tmp_path / f"{stem}.parquet", index=False, engine="pyarrow"
            )

        client.log_artifacts(
            run_id=mlflow_run_id,
            local_dir=str(tmp_path),
            artifact_path=config.mlflow_artifact_path,
        )


if __name__ == "__main__":
    main()
