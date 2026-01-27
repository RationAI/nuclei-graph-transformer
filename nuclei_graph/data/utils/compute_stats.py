import numpy as np
import pandas as pd
from einops import rearrange
from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_scale,
)
from scipy.spatial import KDTree
from tqdm import tqdm


def compute_scale_stats(df: pd.DataFrame, efd_order: int) -> tuple[float, float]:
    log_scales: list[np.ndarray] = []

    print("Computing scale statistics...")
    for nuclei_path in tqdm(df["slide_nuclei_path"]):
        nuclei_df = pd.read_parquet(nuclei_path, columns=["polygon"])

        contours = rearrange(nuclei_df["polygon"].tolist(), "b (v c) -> b v c", c=2)
        efd = elliptic_fourier_descriptors(contours, efd_order)

        _, scales = normalize_efd_for_scale(efd)
        log_scales.append(np.log(np.maximum(scales, 1e-8)))

    scales_combined = np.concatenate(log_scales)
    scale_mean = float(scales_combined.mean())
    scale_std = float(scales_combined.std())
    print(f"Computed scale mean: {scale_mean}, scale std: {scale_std}")

    return scale_mean, scale_std


def compute_median_neighbor_distance(df: pd.DataFrame) -> float:
    all_neighbor_dists: list[np.ndarray] = []

    print("Computing median neighbor distance...")
    for nuclei_path in tqdm(df["slide_nuclei_path"]):
        nuclei_df = pd.read_parquet(nuclei_path, columns=["centroid"])

        coords = np.stack(nuclei_df["centroid"].tolist())
        dists, _ = KDTree(coords).query(coords, k=2)
        nn_dists = dists[:, 1]  # first column is distance to self

        # filter out outliers
        valid_dists = nn_dists[nn_dists < np.percentile(nn_dists, 99)]
        all_neighbor_dists.append(valid_dists)

    combined_dists = np.concatenate(all_neighbor_dists)
    median_dist = float(np.median(combined_dists))
    print("Computed median neighbor distance:", median_dist)

    return median_dist
