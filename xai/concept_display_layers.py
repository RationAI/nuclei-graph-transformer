"""Display layers of the blinded review's concepts for xOpat: nucleus and overview masks.

For each concept in `letters` (`concept_<letter>`, as in `xai.concept_masks`) and each
slide in `slides_csv` (all test slides if null):

- `display_masks/<label>/<slide_id>.tiff` (level 0): the nucleus masks of
  `xai.concept_masks` (v = max(1, round(255 · min(c / s_j, 1))), 0 if c = 0) remapped to
  0 for v ≤ `cutoff`, else round(255 · (v - cutoff) / (255 - cutoff)). Shown as an xOpat
  `heatmap` (alpha ∝ value), this is alpha 0 at the cutoff rising linearly to the layer
  opacity at 255. The heatmap can hide values below a threshold, but its alpha at the
  threshold is the threshold itself, so the ramp has to be in the pixels.
- `overview_masks/<label>/<slide_id>.tiff` (level `overview.level`): activation density
  D_j = Σ_i (c_ij / s_j) · G_sigma(x - x_i) over the nucleus centroids, as
  u = round(255 · min(D_j / T_j, 1)), where T_j is the `scale_quantile` quantile of D_j
  at the active nuclei (c_ij > 0) of all test slides, remapped like the nucleus masks
  with `overview.cutoff` (0 for u ≤ cutoff, then linear to 255).
- `scales.json`: label -> s_j and T_j.

As in `xai.concept_masks`, nothing logged tells a concept's SAE index.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyvips
import ray
from joblib import Parallel, delayed
from omegaconf import DictConfig
from rationai.masks import process_items, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from scipy.ndimage import gaussian_filter

from nuclei_graph.mlflow_utils import tag_parent_run
from xai.concept_masks import code_commit, concept_key, pixel_values, render_slide


def display_values(values: np.ndarray, cutoff: int) -> np.ndarray:
    """0 for v ≤ cutoff, else round(255 · (v - cutoff) / (255 - cutoff))."""
    ramp = np.rint(255 * (values.astype(np.float64) - cutoff) / (255 - cutoff))
    return np.where(values > cutoff, ramp, 0).astype(np.uint8)


def density(
    centroids_um: np.ndarray,
    weights: np.ndarray,
    size: tuple[int, int],
    mpp: float,
    sigma_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian density (concepts x height x width) and each nucleus' pixel (x, y)."""
    width, height = size
    px = np.clip((centroids_um / mpp).astype(np.int64), 0, [width - 1, height - 1])
    grid = np.zeros((weights.shape[1], height, width), dtype=np.float64)
    for j in range(weights.shape[1]):
        np.add.at(grid[j], (px[:, 1], px[:, 0]), weights[:, j])
        grid[j] = gaussian_filter(grid[j], sigma_um / mpp, mode="constant")
    return grid, px


def overview_geometry(
    slide_path: str, level: int
) -> tuple[tuple[int, int], float, float]:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide.slide_resolution(level)
        return slide.level_dimensions[level], mpp_x, mpp_y


def density_at_nuclei(
    slide_path: str,
    centroids_um: np.ndarray,
    weights: np.ndarray,
    level: int,
    sigma_um: float,
) -> np.ndarray:
    size, mpp_x, _ = overview_geometry(slide_path, level)
    grid, px = density(centroids_um, weights, size, mpp_x, sigma_um)
    return grid[:, px[:, 1], px[:, 0]].T.astype(np.float32)


@ray.remote
def render_display_slide(
    item: dict[str, Any],
    labels: list[str],
    scales: np.ndarray,
    overview: DictConfig,
    tile_width: int,
    tile_height: int,
    display_dir: Path,
    overview_dir: Path,
) -> None:
    render_slide(item, labels, 0, tile_width, tile_height, display_dir)

    size, mpp_x, mpp_y = overview_geometry(item["slide_path"], overview.level)
    grid, _ = density(
        item["centroids_um"], item["weights"], size, mpp_x, overview.sigma_um
    )
    name = Path(item["slide_path"]).with_suffix(".tiff").name
    for j, label in enumerate(labels):
        values = np.rint(255 * np.minimum(grid[j] / scales[j], 1.0)).astype(np.uint8)
        values = display_values(values, overview.cutoff)
        path = overview_dir / label / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_big_tiff(
            image=pyvips.Image.new_from_array(values),
            path=path,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            tile_width=tile_width,
            tile_height=tile_height,
        )


@with_cli_args(["+xai=concept_display_layers"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)

    key = concept_key(config.deck_key, list(config.letters))
    labels = key["label"].tolist()

    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == "test"].sort_values("offset")
    index = pq.read_table(
        Path(config.activations_dir) / "test" / "index.parquet",
        columns=["nucleus_id", "centroid_x_um", "centroid_y_um"],
    )
    codes = np.load(Path(config.concept_dir) / "codes_test.npy", mmap_mode="r")
    assert len(index) == len(codes) == slides["n_nuclei"].sum()
    c = np.asarray(codes)[:, key["concept"].to_numpy()]
    scales = np.array(
        [
            np.quantile(c[c[:, j] > 0, j], config.scale_quantile)
            for j in range(c.shape[1])
        ]
    )
    display = display_values(pixel_values(c, scales), config.cutoff)
    weights = (c / scales).astype(np.float32)
    centroids = np.stack(
        [
            index.column("centroid_x_um").to_numpy(),
            index.column("centroid_y_um").to_numpy(),
        ],
        axis=1,
    )

    sections = [
        slice(slide.offset, slide.offset + slide.n_nuclei)
        for slide in slides.itertuples(index=False)
    ]
    sampled = Parallel(n_jobs=config.overview.n_jobs)(
        delayed(density_at_nuclei)(
            slide.slide_path,
            centroids[section],
            weights[section],
            config.overview.level,
            config.overview.sigma_um,
        )
        for slide, section in zip(slides.itertuples(index=False), sections, strict=True)
    )
    d = np.concatenate(sampled)
    overview_scales = np.array(
        [
            np.quantile(d[c[:, j] > 0, j], config.overview.scale_quantile)
            for j in range(c.shape[1])
        ]
    )

    if config.slides_csv is not None:
        selected = set(pd.read_csv(config.slides_csv)["slide_id"])
        slides = slides[slides["slide_id"].isin(selected)]
    nucleus_ids = index.column("nucleus_id")
    items = [
        {
            "slide_id": slide.slide_id,
            "slide_path": slide.slide_path,
            "slide_nuclei_path": slide.slide_nuclei_path,
            "nucleus_ids": nucleus_ids.slice(slide.offset, slide.n_nuclei).to_numpy(),
            "values": display[slide.offset : slide.offset + slide.n_nuclei],
            "centroids_um": centroids[slide.offset : slide.offset + slide.n_nuclei],
            "weights": weights[slide.offset : slide.offset + slide.n_nuclei],
        }
        for slide in slides.itertuples(index=False)
    ]

    with TemporaryDirectory() as tmp:
        display_dir, overview_dir = (
            Path(tmp) / "display_masks",
            Path(tmp) / "overview_masks",
        )
        process_items(
            items=items,
            process_item=render_display_slide.options(
                memory=config.task_memory_gb * 1024**3
            ),
            fn_kwargs={
                "labels": labels,
                "scales": overview_scales,
                "overview": config.overview,
                "tile_width": config.mask_tile_width,
                "tile_height": config.mask_tile_height,
                "display_dir": display_dir,
                "overview_dir": overview_dir,
            },
            max_concurrent=config.max_concurrent,
        )
        (Path(tmp) / "scales.json").write_text(
            json.dumps(
                {
                    label: {"s": float(s), "overview_T": float(t)}
                    for label, s, t in zip(labels, scales, overview_scales, strict=True)
                },
                indent=2,
            )
        )
        commit, dirty = code_commit()
        logger.log_hyperparams(
            {"code_commit": commit, "code_dirty": dirty, "slides": len(items)}
        )
        logger.log_artifacts(str(display_dir), "display_masks")
        logger.log_artifacts(str(overview_dir), "overview_masks")
        logger.log_artifact(str(Path(tmp) / "scales.json"))


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
