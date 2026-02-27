import math
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm


def compute_feature_statistics(
    df: pd.DataFrame, efds_path: str | Path, target_dim: int
) -> tuple[tuple[float, float], dict[str, Tensor]]:
    """Computes global log-scale mean, log-scale std, EFD mean, and EFD standard deviation."""
    df = df.sort_values(by="slide_id").reset_index(drop=True)

    total_log_scale_sum = 0.0
    total_log_scale_sq_sum = 0.0
    total_count = 0

    efd_sum = torch.zeros(target_dim, dtype=torch.float64)
    efd_sq_sum = torch.zeros(target_dim, dtype=torch.float64)

    efds_path = Path(efds_path)
    print(f"[INFO] Computing feature statistics from: {efds_path}")

    for slide_id in tqdm(df["slide_id"], desc="Computing Statistics"):
        file_path = efds_path / f"{slide_id}.pt"
        data = torch.load(file_path, map_location="cpu", weights_only=True)

        # Log-transform the scales for a normal distribution
        scales = data["scales"]
        log_scales = torch.log(scales + 1e-6)

        efds = data["efds"][:, :target_dim].to(torch.float64)

        total_log_scale_sum += log_scales.sum().item()
        total_log_scale_sq_sum += (log_scales**2).sum().item()
        total_count += scales.numel()

        efd_sum += efds.sum(dim=0)
        efd_sq_sum += (efds**2).sum(dim=0)

    # Compute Log-Scale Stats
    log_scale_mean = total_log_scale_sum / total_count
    log_scale_var = (total_log_scale_sq_sum / total_count) - (log_scale_mean**2)
    log_scale_std = math.sqrt(max(log_scale_var, 1e-8))

    # Compute EFD Stats
    efd_mean = efd_sum / total_count
    efd_var = (efd_sq_sum / total_count) - (efd_mean**2)
    efd_std = torch.sqrt(torch.clamp(efd_var, min=1e-8))

    print(f"[INFO] Computed log_scale mean: {log_scale_mean:.4f}")
    print(f"[INFO] Computed log_scale std: {log_scale_std:.4f}")
    print(f"[INFO] Computed EFD mean: {efd_mean}")
    print(f"[INFO] Computed EFD std: {efd_std}")

    return (log_scale_mean, log_scale_std), {
        "mean": efd_mean.float(),
        "std": efd_std.float(),
    }
