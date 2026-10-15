import numpy as np
from numpy.typing import NDArray


def capped_shares(shares: NDArray[np.float64], cap: float) -> NDArray[np.float64]:
    """Water-filling: t_k = min(λ s_k, cap) with Σ t_k = 1 (requires len(shares)·cap ≥ 1)."""
    lo, hi = 1.0, 1.0 / max(shares[shares > 0].min(), 1e-300)
    for _ in range(200):
        lam = (lo + hi) / 2
        if np.minimum(lam * shares, cap).sum() < 1.0:
            lo = lam
        else:
            hi = lam
    return np.minimum(hi * shares, cap)


def balanced_slide_capped_weights(
    labels: NDArray[np.integer],
    slide_codes: NDArray[np.integer],
    max_slide_share: float = 0.01,
    tol: float = 0.01,
    max_iter: int = 100,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Per-nucleus sampling probabilities (summing to 1).

    Carcinoma and benign nuclei get equal total mass, and no slide gets more than
    about `max_slide_share` of it; the two constraints are alternated until both hold
    within `tol` (relative). Within a slide and label, nuclei are sampled uniformly.
    """
    labels = np.asarray(labels)
    slide_codes = np.asarray(slide_codes)
    num_slides = int(slide_codes.max()) + 1
    cap = max(max_slide_share, 1.0 / np.unique(slide_codes).size)
    positive = labels == 1

    weights = np.empty(len(labels), dtype=np.float64)
    for mask in (positive, ~positive):
        if mask.any():
            weights[mask] = 1.0 / mask.sum()
    weights /= weights.sum()

    def label_share() -> float:
        return float(weights[positive].sum())

    for _ in range(max_iter):
        shares = np.bincount(slide_codes, weights=weights, minlength=num_slides)
        present = shares > 0
        target = np.zeros_like(shares)
        target[present] = capped_shares(shares[present], cap)
        scale = np.divide(target, shares, out=np.zeros_like(shares), where=present)
        weights *= scale[slide_codes]

        share = label_share()
        if 0.0 < share < 1.0:
            weights[positive] *= 0.5 / share
            weights[~positive] *= 0.5 / (1.0 - share)
        weights /= weights.sum()

        max_share = np.bincount(slide_codes, weights=weights).max()
        if max_share <= cap * (1 + tol) and abs(label_share() - 0.5) <= 0.5 * tol:
            break

    shares = np.bincount(slide_codes, weights=weights)
    stats = {
        "max_slide_share": float(shares.max()),
        "slide_share_cap": cap,
        "carcinoma_share": label_share(),
        "effective_slides": float(1.0 / np.square(shares).sum()),
        "effective_nuclei": float(1.0 / np.square(weights).sum()),
    }
    return weights, stats


class WeightedIndexSampler:
    """Draws indices with replacement by inverse-CDF sampling (any number of categories)."""

    def __init__(self, weights: NDArray[np.float64], seed: int) -> None:
        self.cdf = np.cumsum(weights, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> NDArray[np.int64]:
        u = self.rng.random(n) * self.cdf[-1]
        return np.minimum(np.searchsorted(self.cdf, u, side="right"), len(self.cdf) - 1)
