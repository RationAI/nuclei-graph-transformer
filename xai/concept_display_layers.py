"""Display layers of the blinded review's concepts for xOpat: nucleus and regions masks.

For each concept in `letters` (`concept_<letter>`, as in `xai.concept_masks`) and each
slide in `slides_csv` (all test slides if null):

- `display_masks/<label>/<slide_id>.tiff` (level 0): the nucleus masks of
  `xai.concept_masks` (v = max(1, round(255 · min(c / s_j, 1))), 0 if c = 0) remapped to
  0 for v ≤ `cutoff`, else round(255 · (v - cutoff) / (255 - cutoff)). Shown as an xOpat
  `heatmap` (alpha ∝ value), this is alpha 0 at the cutoff rising linearly to the layer
  opacity at 255. The heatmap can hide values below a threshold, but its alpha at the
  threshold is the threshold itself, so the ramp has to be in the pixels.
- `regions_masks/<label>/<slide_id>.tiff` (level `regions.level`): the shrunk local
  mean activation m_j = Σ_i c_ij G(x - p_i) / (Σ_i G(x - p_i) + κ) over the nucleus
  centroids p_i, with G a Gaussian of `regions.sigma_um` (discrete, sum 1, cut at 4 sigma) and
  κ the kernel mass of `regions.kappa_nuclei` nuclei within sigma (kappa_nuclei / (π sigma²) per
  µm²), 0 outside the tissue mask. As u = round(255 · min(m_j / Q_j, 1)), where Q_j is the
  `regions.scale_quantile` quantile of m_j over the tissue pixels of all test slides,
  remapped like the nucleus masks with `regions.cutoff` (0 for u ≤ cutoff, then linear
  to 255).
- `scales.json`: label -> s_j and Q_j.

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
from PIL import Image
from rationai.masks import process_items, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from scipy.ndimage import gaussian_filter

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.tissue import TissueMask
from xai.concept_masks import code_commit, concept_key, pixel_values, render_slide


def display_values(values: np.ndarray, cutoff: int) -> np.ndarray:
    """0 for v ≤ cutoff, else round(255 · (v - cutoff) / (255 - cutoff))."""
    ramp = np.rint(255 * (values.astype(np.float64) - cutoff) / (255 - cutoff))
    return np.where(values > cutoff, ramp, 0).astype(np.uint8)


def regions_geometry(
    slide_path: str, level: int
) -> tuple[tuple[int, int], float, float]:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide.slide_resolution(level)
        return slide.level_dimensions[level], mpp_x, mpp_y


def local_mean(
    centroids_um: np.ndarray,
    c: np.ndarray,
    size: tuple[int, int],
    mpp: float,
    sigma_um: float,
    kappa_nuclei: float,
) -> np.ndarray:
    """Shrunk local mean Σ_i c_ij G / (Σ_i G + κ), concepts x height x width."""
    width, height = size
    sigma = sigma_um / mpp
    px = np.clip((centroids_um / mpp).astype(np.int64), 0, [width - 1, height - 1])
    count = np.zeros((height, width))
    np.add.at(count, (px[:, 1], px[:, 0]), 1.0)
    count = gaussian_filter(count, sigma, mode="constant")
    # κ in the units of `count` (kernel mass per pixel): kappa_nuclei / (πσ²) per µm²
    kappa = kappa_nuclei / (np.pi * sigma_um**2) * mpp**2
    mean = np.empty((c.shape[1], height, width), dtype=np.float32)
    for j in range(c.shape[1]):
        total = np.zeros((height, width))
        np.add.at(total, (px[:, 1], px[:, 0]), c[:, j])
        mean[j] = gaussian_filter(total, sigma, mode="constant") / (count + kappa)
    return mean


def tissue_grid(
    path: str, mpp0: float, mpp: float, size: tuple[int, int]
) -> np.ndarray:
    """The tissue mask resampled (nearest) to the regions grid."""
    mask = TissueMask.read(path, mpp0, mpp).mask
    return np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST))


def slide_regions(item: dict[str, Any], regions: DictConfig) -> tuple[np.ndarray, Any]:
    """Regions values (0 outside tissue), the tissue grid and the grid's (size, mpp)."""
    size, mpp_x, mpp_y = regions_geometry(item["slide_path"], regions.level)
    mean = local_mean(
        item["centroids_um"],
        item["c"],
        size,
        mpp_x,
        regions.sigma_um,
        regions.kappa_nuclei,
    )
    tissue = tissue_grid(item["tissue_path"], item["mpp0"], mpp_x, size)
    mean[:, ~tissue] = 0.0
    return mean, (tissue, size, mpp_x, mpp_y)


def regions_histogram(
    item: dict[str, Any], regions: DictConfig, edges: list[np.ndarray]
) -> np.ndarray:
    """Per concept, the histogram of the regions values over the slide's tissue pixels."""
    mean, (tissue, *_) = slide_regions(item, regions)
    return np.stack(
        [np.histogram(mean[j][tissue], bins=edges[j])[0] for j in range(len(edges))]
    )


@ray.remote
def render_display_slide(
    item: dict[str, Any],
    labels: list[str],
    scales: np.ndarray,
    regions: DictConfig,
    tile_width: int,
    tile_height: int,
    display_dir: Path,
    regions_dir: Path,
) -> None:
    render_slide(item, labels, 0, tile_width, tile_height, display_dir)

    mean, (_, _, mpp_x, mpp_y) = slide_regions(item, regions)
    name = Path(item["slide_path"]).with_suffix(".tiff").name
    for j, label in enumerate(labels):
        values = np.rint(255 * np.minimum(mean[j] / scales[j], 1.0)).astype(np.uint8)
        values = display_values(values, regions.cutoff)
        path = regions_dir / label / name
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
    centroids = np.stack(
        [
            index.column("centroid_x_um").to_numpy(),
            index.column("centroid_y_um").to_numpy(),
        ],
        axis=1,
    )

    items = [
        {
            "slide_id": slide.slide_id,
            "slide_path": slide.slide_path,
            "slide_nuclei_path": slide.slide_nuclei_path,
            "tissue_path": str(
                Path(config.regions.tissue_masks) / f"{slide.slide_id}.tiff"
            ),
            "mpp0": slide.mpp_x,
            "offset": slide.offset,
            "values": display[slide.offset : slide.offset + slide.n_nuclei],
            "centroids_um": centroids[slide.offset : slide.offset + slide.n_nuclei],
            "c": c[slide.offset : slide.offset + slide.n_nuclei],
        }
        for slide in slides.itertuples(index=False)
    ]
    # Q_j from a fine histogram over [0, max c_j] (the local mean cannot exceed it)
    edges = [
        np.linspace(0.0, float(c[:, j].max()), config.regions.bins + 1)
        for j in range(c.shape[1])
    ]
    histogram = sum(
        Parallel(n_jobs=config.regions.n_jobs)(
            delayed(regions_histogram)(item, config.regions, edges) for item in items
        )
    )
    cumulative = np.cumsum(histogram, axis=1) / histogram.sum(axis=1, keepdims=True)
    regions_scales = np.array(
        [
            edges[j][np.searchsorted(cumulative[j], config.regions.scale_quantile) + 1]
            for j in range(c.shape[1])
        ]
    )

    if config.slides_csv is not None:
        selected = set(pd.read_csv(config.slides_csv)["slide_id"])
        items = [item for item in items if item["slide_id"] in selected]
    nucleus_ids = index.column("nucleus_id")
    for item in items:
        item["nucleus_ids"] = nucleus_ids.slice(
            item["offset"], len(item["c"])
        ).to_numpy()

    with TemporaryDirectory() as tmp:
        display_dir, regions_dir = (
            Path(tmp) / "display_masks",
            Path(tmp) / "regions_masks",
        )
        process_items(
            items=items,
            process_item=render_display_slide.options(
                memory=config.task_memory_gb * 1024**3
            ),
            fn_kwargs={
                "labels": labels,
                "scales": regions_scales,
                "regions": config.regions,
                "tile_width": config.mask_tile_width,
                "tile_height": config.mask_tile_height,
                "display_dir": display_dir,
                "regions_dir": regions_dir,
            },
            max_concurrent=config.max_concurrent,
        )
        (Path(tmp) / "scales.json").write_text(
            json.dumps(
                {
                    label: {"s": float(s), "regions_Q": float(q)}
                    for label, s, q in zip(labels, scales, regions_scales, strict=True)
                },
                indent=2,
            )
        )
        commit, dirty = code_commit()
        logger.log_hyperparams(
            {"code_commit": commit, "code_dirty": dirty, "slides": len(items)}
        )
        logger.log_artifacts(str(display_dir), "display_masks")
        logger.log_artifacts(str(regions_dir), "regions_masks")
        logger.log_artifact(str(Path(tmp) / "scales.json"))


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
