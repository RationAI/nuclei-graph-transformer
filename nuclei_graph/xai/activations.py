from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class Activations:
    """Outputs of `ActivationsCallback` for one split, aligned row by row.

    `h` and `logits` are memory-mapped; slides are contiguous in `index`.
    """

    h: NDArray[np.floating]  # (n, d), output of `final_norm` (float16)
    logits: NDArray[np.floating]  # (n,), model nucleus logits
    index: pd.DataFrame  # slide_id, nucleus_id, label, centroid_x_um, centroid_y_um

    def __len__(self) -> int:
        return len(self.index)

    @property
    def labels(self) -> NDArray[np.int64]:
        return self.index["label"].to_numpy(np.int64)

    @property
    def slide_codes(self) -> NDArray[np.int64]:
        return self.index["slide_id"].astype("category").cat.codes.to_numpy(np.int64)

    def z(self, rows: slice | NDArray[np.integer]) -> NDArray[np.float64]:
        return np.asarray(self.h[rows], dtype=np.float64)

    def chunks(self, chunk_size: int = 262144) -> Iterator[slice]:
        for start in range(0, len(self), chunk_size):
            yield slice(start, min(start + chunk_size, len(self)))


def load_activations(activations_dir: str | Path, split: str) -> Activations:
    split_dir = Path(activations_dir) / split
    activations = Activations(
        h=np.load(split_dir / "h.npy", mmap_mode="r"),
        logits=np.load(split_dir / "logits.npy", mmap_mode="r"),
        index=pd.read_parquet(split_dir / "index.parquet"),
    )
    assert len(activations.h) == len(activations.logits) == len(activations.index)
    return activations


def holdout_slides(
    slide_ids: pd.Series, holdout_size: float, seed: int
) -> NDArray[np.bool_]:
    """Row mask of a random `holdout_size` fraction of the slides."""
    slides = np.sort(slide_ids.astype(str).unique())
    rng = np.random.default_rng(seed)
    held = rng.choice(
        slides, size=max(1, round(holdout_size * len(slides))), replace=False
    )
    return slide_ids.astype(str).isin(set(held)).to_numpy()
