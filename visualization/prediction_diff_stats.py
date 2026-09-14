"""Script for computing normalized disagreement statistics from prediction-diff masks.

Reads the already-rasterized per-slide prediction-diff masks produced by
`prediction_diff_mask.py` (0 = background/agreement, 255 = disagreement) and,
for each slide, divides the disagreement area by the foreground area of a
reference mask (e.g. the annotated/tumor region mask). Slides' annotated
regions vary widely in size, so raw disagreement pixel counts are not
comparable across slides; the normalized ratio is.

Masks are matched between the diff and normalization directories by filename
(both are named after the slide's stem, see `prediction_diff_mask.py` and
`preprocessing/annotation_masks.py`).
"""

from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import pyvips
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


def mask_foreground_area(mask_path: Path) -> int:
    """Counts foreground (255) pixels in a single-channel binary mask TIFF."""
    image = pyvips.Image.new_from_file(str(mask_path), access="sequential")
    fraction_foreground = image.avg() / 255.0
    return round(fraction_foreground * image.width * image.height)


@ray.remote(memory=4 * 1024**3)
def compute_slide_stats(
    item: dict[str, Any],
    diff_masks_dir: Path,
    normalization_masks_dir: Path,
) -> dict[str, Any] | None:
    slide_path = Path(item["slide_path"])
    mask_name = slide_path.with_suffix(".tiff").name

    diff_mask_path = diff_masks_dir / mask_name
    normalization_mask_path = normalization_masks_dir / mask_name
    if not diff_mask_path.exists() or not normalization_mask_path.exists():
        return None

    normalization_area = mask_foreground_area(normalization_mask_path)
    if normalization_area == 0:
        return None

    disagreement_area = mask_foreground_area(diff_mask_path)
    return {
        "slide_id": item.get("slide_id", slide_path.stem),
        "disagreement_area_px": disagreement_area,
        "normalization_area_px": normalization_area,
        "disagreement_ratio": disagreement_area / normalization_area,
    }


def uris2df(uris: list[str]) -> pd.DataFrame:
    """Loads and merges multiple metadata Parquet files into a single DataFrame."""
    batches = [pd.read_parquet(download_artifacts(uri)) for uri in uris]
    return pd.concat(batches, ignore_index=True).drop_duplicates(subset=["slide_path"])


@hydra.main(config_path="../configs", config_name="prediction_diff_stats", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata = uris2df(config.metadata_uris)
    diff_masks_dir = Path(download_artifacts(config.diff_masks_uri))
    normalization_masks_dir = Path(download_artifacts(config.normalization_masks_uri))

    items = metadata.to_dict("records")
    pending = [
        compute_slide_stats.remote(item, diff_masks_dir, normalization_masks_dir)
        for item in items
    ]
    results = [result for result in ray.get(pending) if result is not None]

    print(f"Computed disagreement stats for {len(results)} / {len(items)} slides.\n")

    stats = pd.DataFrame(results).sort_values("disagreement_ratio", ascending=False)

    print(f"Top {config.top_n} slides with the HIGHEST normalized disagreement:")
    print(stats.head(config.top_n).to_string(index=False))

    print(f"\nTop {config.top_n} slides with the LOWEST normalized disagreement:")
    print(stats.tail(config.top_n)[::-1].to_string(index=False))

    output_path = Path(f"{config.mlflow_artifact_path}.csv")
    stats.to_csv(output_path, index=False)
    logger.log_artifact(str(output_path))
    output_path.unlink()


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
