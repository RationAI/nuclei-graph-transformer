import numpy as np
import pandas as pd
from einops import rearrange
from scipy.spatial import KDTree
from tqdm import tqdm

from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_scale,
)


def compute_scale_stats(df: pd.DataFrame, efd_order: int) -> tuple[float, float]:
    log_scales = []

    print("Computing scale statistics...")
    for nuclei_path in tqdm(df["slide_nuclei_path"]):
        nuclei = pd.read_parquet(nuclei_path, columns=["polygon"])
        contours = rearrange(nuclei["polygon"].tolist(), "b (v c) -> b v c", c=2)

        efd = elliptic_fourier_descriptors(contours, efd_order)
        _, scales = normalize_efd_for_scale(efd)
        log_scales.append(np.log(np.maximum(scales, 1e-8)))

    log_scales = np.concatenate(log_scales)

    scale_mean = float(log_scales.mean())
    scale_std = float(log_scales.std())
    print(f"Computed scale mean: {scale_mean}, scale std: {scale_std}")

    return scale_mean, scale_std


def compute_average_neighbor_distance(df: pd.DataFrame) -> float:
    all_neighbor_dists = []

    print("Computing average neighbor distance...")
    for nuclei_path in tqdm(df["slide_nuclei_path"]):
        nuclei = pd.read_parquet(nuclei_path, columns=["centroid"])
        coords = np.stack(nuclei["centroid"].tolist())

        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=2)
        nn_dists = dists[:, 1]  # first column is distance to self

        # filter out outliers
        valid_dists = nn_dists[nn_dists < np.percentile(nn_dists, 99)]
        all_neighbor_dists.append(valid_dists)

    combined_dists = np.concatenate(all_neighbor_dists)
    median_dist = float(np.median(combined_dists))
    print("Computed average neighbor distance:", median_dist)

    return median_dist
