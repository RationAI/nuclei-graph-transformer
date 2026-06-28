import pandas as pd
import torch
from torch import Tensor

from nuclei_graph.data.datasets.tile.base import BaseTileDataset, get_slide_data


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
    """Pools per-nucleus predictions into a single prediction for each tile of a slide.

    For every tile, nuclei centroids are first matched to the full tile area, then
    restricted to the tile's inner ROI. Only nuclei that fall inside this inner ROI
    are pooled into the tile's prediction.

    Tiles whose inner ROI contains no nuclei are assigned a prediction of 0.0.

    Args:
        tiles: Rows of tile metadata with "x" and "y" tile coordinates.
        dataset: Tile dataset providing tile geometry helpers.
        nuclei_preds: Per-nucleus predictions indexed by nucleus id, for the
            slide that `tiles` belongs to.
        pooling_mode: One of `POOLING_MODES` ("max", "mean", "top_k") used to
            aggregate the nuclei predictions within a tile's ROI.
        k: Number of top predictions to average when `pooling_mode` is "top_k".
            Ignored otherwise.

    Returns:
        A DataFrame with one row per tile and columns "x", "y", and
        "tile_prediction" (the pooled score for that tile's ROI).
    """
    stem = tiles["stem"].iloc[0]
    nuclei_path = dataset.slide_props[stem]["slide_nuclei_path"]
    _, centroids, centroid_tree, nuclei_ids = get_slide_data(nuclei_path)

    rows = []
    for _, tile in tiles.iterrows():
        scaled_props = dataset.get_scaled_props(tile)
        tile_indices = dataset.get_tile_indices(scaled_props, centroids, centroid_tree)

        tile_pred = 0.0
        if len(tile_indices) > 0:
            roi_mask = dataset.get_roi_mask(scaled_props, centroids[tile_indices])
            roi_indices = tile_indices[roi_mask]

            if len(roi_indices) > 0:
                tile_ids = nuclei_ids[roi_indices]
                valid_preds = torch.as_tensor(
                    nuclei_preds.loc[tile_ids].to_numpy(), dtype=torch.float32
                )
                tile_pred = pool_predictions(valid_preds, pooling_mode, k).item()

        rows.append(
            {"x": int(tile["x"]), "y": int(tile["y"]), "tile_prediction": tile_pred}
        )

    return pd.DataFrame(rows)
