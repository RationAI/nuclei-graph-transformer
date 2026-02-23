from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm


def compute_feature_statistics(
    df: pd.DataFrame, efds_path: str | Path, target_dim: int
) -> tuple[float, Tensor, Tensor]:
    """Computes global scale mean, EFD mean, and EFD standard deviation."""
    total_scale_sum = 0.0
    total_count = 0

    efd_sum = torch.zeros(target_dim, dtype=torch.float64)
    efd_sq_sum = torch.zeros(target_dim, dtype=torch.float64)

    efds_path = Path(efds_path)
    print(f"[INFO] Computing feature statistics from: {efds_path}")

    for slide_id in tqdm(df["slide_id"], desc="Computing Statistics"):
        file_path = efds_path / f"{slide_id}.pt"
        data = torch.load(file_path, map_location="cpu", weights_only=True)

        scales = data["scales"]
        efds = data["efds"][:, :target_dim].to(torch.float64)

        total_scale_sum += scales.sum().item()
        total_count += scales.numel()

        efd_sum += efds.sum(dim=0)
        efd_sq_sum += (efds**2).sum(dim=0)

    scale_mean = float(total_scale_sum / total_count)
    efd_mean = efd_sum / total_count
    efd_var = (efd_sq_sum / total_count) - (efd_mean**2)
    efd_std = torch.sqrt(torch.clamp(efd_var, min=1e-8))

    print(f"[INFO] Computed scale mean: {scale_mean:.4f}")
    print(f"[INFO] Computed EFD mean: {efd_mean}")
    print(f"[INFO] Computed EFD std: {efd_std}")
    return scale_mean, efd_mean.float(), efd_std.float()
