from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def compute_scale_mean(df: pd.DataFrame, efds_path: str | Path) -> float:
    total_sum = 0.0
    total_count = 0

    efds_path = Path(efds_path)
    print(f"Computing scale statistics from features in: {efds_path}")

    for slide_id in tqdm(df["slide_id"], desc="Loading scales"):
        file_path = efds_path / f"{slide_id}.pt"

        data = torch.load(file_path, map_location="cpu", weights_only=True)
        scales = data["scales"]

        total_sum += scales.sum().item()
        total_count += scales.numel()

    scale_mean = float(total_sum / total_count)
    print(f"Computed scale mean: {scale_mean:.4f}")
    return scale_mean
