"""Shared loaders for tile-level prediction artifacts used by other postprocessing scripts."""

from pathlib import Path

import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig


def get_predictions(predictions_dir: Path) -> pd.DataFrame:
    """Concatenates per-slide pooled-tile-prediction parquet files into a single DataFrame."""
    all_preds = []
    for parquet_path in predictions_dir.rglob("*.parquet"):
        slide_pred_df = pd.read_parquet(parquet_path)
        slide_pred_df["slide_id"] = parquet_path.stem
        all_preds.append(slide_pred_df)
    return pd.concat(all_preds, ignore_index=True)


def load_tile_predictions(config: DictConfig) -> pd.DataFrame:
    """Loads pooled tile predictions and merges them with ground-truth tile labels."""
    tiles_df = pd.read_parquet(download_artifacts(config.tiles_uri))
    slides_df = pd.read_parquet(download_artifacts(config.slides_uri))

    id_to_stem = dict(zip(slides_df["id"], slides_df["stem"], strict=True))
    tiles_df["slide_id"] = tiles_df["slide_id"].map(id_to_stem)
    tiles_df["carcinoma"] = (
        tiles_df["carcinoma_roi_percentage"] > config.carcinoma_roi_t
    ).astype(int)

    predictions_dir = Path(download_artifacts(config.tile_predictions_uri))
    preds_df = get_predictions(predictions_dir)
    merged_df = pd.merge(preds_df, tiles_df, on=["slide_id", "x", "y"], how="inner")

    target_col = config.label_column
    merged_df[target_col] = merged_df[target_col].fillna(0).astype(int)
    return merged_df
