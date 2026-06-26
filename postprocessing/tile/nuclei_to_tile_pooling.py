from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch import Tensor

from nuclei_graph.data.datamodules.base import METADATA_COLS_EVAL
from nuclei_graph.data.datasets.tile.base import BaseTileDataset, get_slide_data
from nuclei_graph.mlflow_utils import tag_parent_run
from postprocessing.mlflow_utils import setup_mlflow


POOLING_MODES = ("max", "mean", "top_k")


def pool_predictions(preds: Tensor, mode: str, k: int = 10) -> Tensor:
    """Pools a 1D tensor of nuclei-level predictions into a single tile/graph score."""
    assert mode in POOLING_MODES, "Invalid pooling mode."
    if mode == "max":
        return preds.max()
    if mode == "mean":
        return preds.mean()
    actual_k = min(k, len(preds))
    top_k_preds, _ = torch.topk(preds, actual_k)
    return top_k_preds.mean()


def pool_slide_tiles(
    tiles: pd.DataFrame,
    dataset: BaseTileDataset,
    nuclei_preds: pd.Series,
    pooling_mode: str,
    k: int,
) -> pd.DataFrame:
    stem = tiles["stem"].iloc[0]
    nuclei_path = dataset.slide_props[stem]["slide_nuclei_path"]
    _, centroids, centroid_tree, nuclei_ids = get_slide_data(nuclei_path)

    rows = []
    for _, tile in tiles.iterrows():
        scaled_props = dataset.get_scaled_props(tile)
        tile_indices = dataset.get_tile_indices(scaled_props, centroids, centroid_tree)

        tile_pred = 0.0
        if len(tile_indices) > 0:
            tile_ids = nuclei_ids[tile_indices]
            valid_preds = torch.as_tensor(
                nuclei_preds.loc[tile_ids].to_numpy(), dtype=torch.float32
            )
            tile_pred = pool_predictions(valid_preds, pooling_mode, k).item()

        rows.append(
            {"x": int(tile["x"]), "y": int(tile["y"]), "tile_prediction": tile_pred}
        )

    return pd.DataFrame(rows)


@with_cli_args(["+postprocessing=tile/nuclei_to_tile_pooling"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
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
                tiles, dataset, nuclei_preds, config.pooling_mode, config.k
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
