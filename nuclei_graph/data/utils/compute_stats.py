import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm

from nuclei_graph.data.efd import (
    elliptic_fourier_descriptors,
    normalize_efd_for_scale,
)


def compute_scale_mean(df: pd.DataFrame, efd_order: int) -> float:
    total_sum = 0.0
    total_count = 0

    print("Computing scale statistics...")
    for nuclei_path in tqdm(df["slide_nuclei_path"]):
        nuclei_df = pd.read_parquet(nuclei_path, columns=["polygon"])

        contours = rearrange(nuclei_df["polygon"].tolist(), "b (v c) -> b v c", c=2)
        efd = elliptic_fourier_descriptors(contours, efd_order)
        _, scales = normalize_efd_for_scale(efd)

        total_sum += np.sum(scales)
        total_count += len(scales)

    if total_count == 0:
        return 0.0

    scale_mean = float(total_sum / total_count)
    print(f"Computed scale mean: {scale_mean:.4f}")

    return scale_mean
