"""Tissue context of every nucleus of a split (step 1b of the concept morphometry).

Writes `context_<split>.parquet` to `output_dir`, one row per nucleus in the order of
`activations/<split>/index.parquet` (row i matches row i of the codes):

- `compartment`: the PANDA Radboud label mask (channel 0: 0 background / unannotated,
  1 stroma, 2 benign epithelium, 3-5 cancerous epithelium, Gleason 3-5) at the nucleus's
  centroid, read at level 0 (the WSI's pixel grid, as the dense nucleus labels);
- `boundary_dist_um`: signed distance to the epithelium-stroma boundary, negative inside
  epithelium (mask >= 2), positive in stroma (mask = 1), NaN on background (0) or if the
  slide lacks either side. Euclidean distance transform of mask level `mask_level` (about
  1.94 um/px), sampled at the centroid's pixel; a pixel's distance to the nearest pixel of
  the other side minus half a pixel (the boundary lies between the two);
- `knn8_mean_dist_um`: mean distance to the 8 nearest other nuclei of the same slide
  (all nuclei; centroids in um from the index);
- from the segmentation polygons (sorted by `id`, checked against the index's id checksum;
  um): `solidity` = area / convex-hull area and `irregularity` = 1 - solidity;
  `pleomorphism_cv` = coefficient of variation (sample std / mean) of the area over the
  8 nearest neighbours (without the nucleus itself; `pleomorphism_incl_self`: with it);
  `orientation_coherence` = mean cos 2(psi_i - psi_j)
  over the 8 nearest neighbours j, psi the long-axis angle of the moment ellipse, for nuclei
  with axis ratio > `min_axis_ratio` and over the neighbours above it too (a round
  nucleus has no defined orientation); NaN otherwise. 1 = all parallel, 0 = random,
  -1 = all perpendicular.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
import tifffile
import zarr
from joblib import Parallel, delayed
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from xai.morphometry_features import polygon_features
from xai.prepare_nuclei import ids_checksum


def mask_level(path: Path, level: int) -> zarr.Array:
    store = tifffile.imread(path, aszarr=True, level=level)
    return zarr.open(store, mode="r")


def signed_boundary_distance(mask: NDArray[np.uint8], pixel_um: float) -> NDArray[np.float32]:
    """Per pixel: -(distance to stroma) in epithelium, +(distance to epithelium) in stroma."""
    epithelium, stroma = mask >= 2, mask == 1
    out = np.full(mask.shape, np.nan, dtype=np.float32)
    if epithelium.any() and stroma.any():
        out[epithelium] = -(distance_transform_edt(~stroma)[epithelium] - 0.5) * pixel_um
        out[stroma] = (distance_transform_edt(~epithelium)[stroma] - 0.5) * pixel_um
    return out


def neighbourhood_shape(polygons_um: NDArray[np.float64], neighbours: NDArray[np.int64],
                        min_axis_ratio: float) -> dict[str, NDArray[np.float32]]:
    """Solidity, irregularity, pleomorphism and orientation coherence (`neighbours`: n x k)."""
    features = polygon_features(polygons_um)
    hulls = shapely.convex_hull(shapely.polygons(polygons_um))
    solidity = features["area_um2"] / shapely.area(hulls)
    areas = features["area_um2"][neighbours]
    pleomorphism = areas.std(axis=1, ddof=1) / areas.mean(axis=1)
    with_self = np.concatenate([features["area_um2"][:, None], areas], axis=1)
    pleomorphism_incl_self = with_self.std(axis=1, ddof=1) / with_self.mean(axis=1)
    elongated = features["axis_ratio"] > min_axis_ratio
    psi = features["orientation_rad"]
    agreement = np.cos(2 * (psi[:, None] - psi[neighbours]))
    use = elongated[neighbours]
    counts = use.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = np.where(use, agreement, 0.0).sum(axis=1) / counts
    coherence[~elongated | (counts == 0)] = np.nan
    return {"solidity": solidity.astype(np.float32), "irregularity": np.maximum(1 - solidity, 0.0).astype(np.float32),
            "pleomorphism_cv": pleomorphism.astype(np.float32),
            "pleomorphism_incl_self": pleomorphism_incl_self.astype(np.float32),
            "orientation_coherence": coherence.astype(np.float32)}


def process_slide(slide: dict, centroids_um: NDArray[np.float64], masks_dir: str, level: int,
                  k: int, min_axis_ratio: float) -> dict:
    mpp = np.array([slide["mpp_x"], slide["mpp_y"]])
    centroids_px = centroids_um / mpp  # level-0 px, pixel i centred at i
    path = Path(masks_dir) / f"{slide['slide_id']}_mask.tiff"

    level0 = mask_level(path, 0)
    height, width = level0.shape[:2]
    x = np.clip(np.floor(centroids_px[:, 0] + 0.5).astype(np.int64), 0, width - 1)
    y = np.clip(np.floor(centroids_px[:, 1] + 0.5).astype(np.int64), 0, height - 1)
    compartment = np.asarray(level0.vindex[y, x, np.zeros_like(x)], dtype=np.int8)

    coarse = np.asarray(mask_level(path, level)[..., 0])
    scale = np.array([coarse.shape[1] / width, coarse.shape[0] / height])
    pixel_um = float(mpp[0] / scale[0])
    distance = signed_boundary_distance(coarse, pixel_um)
    cx = np.clip(np.floor(centroids_px[:, 0] * scale[0]).astype(np.int64), 0, coarse.shape[1] - 1)
    cy = np.clip(np.floor(centroids_px[:, 1] * scale[1]).astype(np.int64), 0, coarse.shape[0] - 1)

    distances, neighbours = cKDTree(centroids_um).query(centroids_um, k=k + 1)  # the first is the nucleus

    nuclei = pd.read_parquet(slide["slide_nuclei_path"], columns=["id", "polygon"]).sort_values("id")
    assert ids_checksum(nuclei["id"].to_numpy()) == slide["ids_crc32"], slide["slide_id"]
    polygons_um = np.array(nuclei["polygon"].tolist(), dtype=np.float64).reshape(len(nuclei), -1, 2) * mpp
    shape = neighbourhood_shape(polygons_um, neighbours[:, 1:], min_axis_ratio)
    return {"compartment": compartment, "boundary_dist_um": distance[cy, cx],
            "knn8_mean_dist_um": distances[:, 1:].mean(axis=1).astype(np.float32), **shape,
            "coarse_compartment": coarse[cy, cx].astype(np.int8), "pixel_um": pixel_um}


@with_cli_args(["+xai=morphometry_context"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
def main(config: DictConfig) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = config.split
    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == split].sort_values("offset").reset_index(drop=True)
    index = pq.read_table(Path(config.activations_dir) / split / "index.parquet",
                          columns=["slide_id", "centroid_x_um", "centroid_y_um"])
    centroids = np.stack([index.column("centroid_x_um").to_numpy(),
                          index.column("centroid_y_um").to_numpy()], axis=1).astype(np.float64)
    assert len(centroids) == slides["n_nuclei"].sum()
    slide_ids = index.column("slide_id").to_numpy()
    starts, ends = slides["offset"].to_numpy(), (slides["offset"] + slides["n_nuclei"] - 1).to_numpy()
    assert (slide_ids[starts] == slides["slide_id"]).all() and (slide_ids[ends] == slides["slide_id"]).all()

    results = Parallel(n_jobs=config.n_jobs, verbose=5)(
        delayed(process_slide)(s, centroids[s["offset"]: s["offset"] + s["n_nuclei"]], config.masks_dir,
                               config.mask_level, config.k, config.min_axis_ratio)
        for s in slides.to_dict("records")
    )
    table = {name: np.concatenate([r[name] for r in results])
             for name in ("compartment", "boundary_dist_um", "knn8_mean_dist_um", "solidity", "irregularity",
                          "pleomorphism_cv", "pleomorphism_incl_self", "orientation_coherence")}
    coarse = np.concatenate([r["coarse_compartment"] for r in results])
    pd.DataFrame(table).to_parquet(out_dir / f"context_{split}.parquet", index=False)

    side = lambda c: np.select([c == 0, c == 1, c >= 2], [0, 1, 2])
    known = (table["compartment"] > 0) & (coarse > 0)
    summary = {
        "split": split, "n_nuclei": len(centroids), "n_slides": len(slides), "k": config.k,
        "mask_level": config.mask_level, "mask_level_pixel_um": sorted({round(r["pixel_um"], 4) for r in results}),
        "compartment_counts": {int(c): int(n) for c, n in zip(*np.unique(table["compartment"], return_counts=True),
                                                             strict=True)},
        "boundary_dist_nan": int(np.isnan(table["boundary_dist_um"]).sum()),
        "orientation_coherence_nan": int(np.isnan(table["orientation_coherence"]).sum()),
        "min_axis_ratio": config.min_axis_ratio,
        "level0_vs_level1_side_agreement": float((side(table["compartment"]) == side(coarse))[known].mean()),
    }
    (out_dir / f"context_{split}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
