"""Completeness metrics and exact concept attributions for affine concept decoders.

The model is split as f = g ∘ h, with z = h(x) and an affine head g(z) = wᵀz + β.
For a concept autoencoder with an affine decoder D(c) = Uᵀc + b_D, gamma = g ∘ D is
affine, so the reconstructed model f_A = gamma(E(z)) decomposes exactly into
b + Σ_j φ_j with φ_j = (wᵀu_j) c_j and b = gamma(0) = wᵀb_D + β.

All computations are in float64; RMSEs over vectors use per-vector norms,
RMSE(a, b) = sqrt(mean_i ‖a_i - b_i‖²), so that FE ≤ ‖w‖₂ · RE holds by Cauchy-Schwarz.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


type Array = NDArray[np.float64]

RIDGE_ALPHAS = tuple(np.logspace(-8, 8, 33))


@dataclass(frozen=True)
class LinearHead:
    """Binary linear head g(z) = wᵀz + β (the model's `class_head`)."""

    w: Array  # (d,)
    beta: float

    def __call__(self, z: Array) -> Array:
        return np.asarray(z, dtype=np.float64) @ self.w + self.beta

    @property
    def lipschitz(self) -> float:
        return float(np.linalg.norm(self.w))

    @classmethod
    def from_npz(cls, path: str) -> "LinearHead":
        head = np.load(path)
        return cls(head["w"].astype(np.float64), float(head["beta"]))


def decode(codes: Array, decoder_weight: Array, decoder_bias: Array) -> Array:
    return codes @ decoder_weight + decoder_bias


def concept_logit_weights(decoder_weight: Array, head: LinearHead) -> Array:
    """wᵀu_j for every concept j; its sign is the concept's direction of effect."""
    return decoder_weight @ head.w


def contributions(codes: Array, decoder_weight: Array, head: LinearHead) -> Array:
    """Exact concept contributions φ_j = (wᵀu_j) c_j, shape (n, k)."""
    return codes * concept_logit_weights(decoder_weight, head)


def baseline(decoder_bias: Array, head: LinearHead) -> float:
    """B = gamma(0) = wᵀb_D + β."""
    return float(head(decoder_bias))


def reconstructed_logits(
    codes: Array, decoder_weight: Array, decoder_bias: Array, head: LinearHead
) -> Array:
    """f_A = g(D(c))."""
    return head(decode(codes, decoder_weight, decoder_bias))


# --- model-agnostic attributions of gamma(c), used to verify that all equal φ_j ---


def insertion(gamma: Callable[[Array], Array], codes: Array) -> Array:
    """gamma(c_j e_j) - gamma(0) for every concept j."""
    out = np.empty_like(codes)
    base = gamma(np.zeros_like(codes))
    for j in range(codes.shape[1]):
        only_j = np.zeros_like(codes)
        only_j[:, j] = codes[:, j]
        out[:, j] = gamma(only_j) - base
    return out


def occlusion(gamma: Callable[[Array], Array], codes: Array) -> Array:
    """gamma(c) - gamma(c with c_j = 0) for every concept j."""
    out = np.empty_like(codes)
    full = gamma(codes)
    for j in range(codes.shape[1]):
        without_j = codes.copy()
        without_j[:, j] = 0.0
        out[:, j] = full - gamma(without_j)
    return out


def grad_times_input(
    gamma: Callable[[torch.Tensor], torch.Tensor], codes: Array
) -> Array:
    """∂gamma/∂c_j · c_j, by autograd."""
    c = torch.tensor(codes, dtype=torch.float64, requires_grad=True)
    gamma(c).sum().backward()
    assert c.grad is not None
    return (c.grad * c).detach().numpy()


# --- heads from codes to logits (η in MCE) ---


class Regressor(Protocol):
    def predict(self, x: Array) -> Array: ...


class RidgeStatistics:
    """Sufficient statistics of a linear regression, accumulated over chunks."""

    def __init__(self, num_features: int) -> None:
        self.n = 0
        self.sx = np.zeros(num_features)
        self.sy = 0.0
        self.sxx = np.zeros((num_features, num_features))
        self.sxy = np.zeros(num_features)

    def update(self, x: Array, y: Array) -> None:
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        self.n += len(x)
        self.sx += x.sum(0)
        self.sy += y.sum()
        self.sxx += x.T @ x
        self.sxy += x.T @ y


@dataclass(frozen=True)
class RidgeHead:
    """η(c) = coefᵀc + intercept; the intercept is not penalized."""

    coef: Array
    intercept: float
    alpha: float

    def predict(self, x: Array) -> Array:
        return np.asarray(x, np.float64) @ self.coef + self.intercept

    @classmethod
    def from_statistics(cls, stats: RidgeStatistics, alpha: float) -> "RidgeHead":
        mean_x, mean_y = stats.sx / stats.n, stats.sy / stats.n
        cxx = stats.sxx - stats.n * np.outer(mean_x, mean_x)
        cxy = stats.sxy - stats.n * mean_x * mean_y
        # least squares: the minimum-norm solution if a code is constant (e.g. dead)
        coef = np.linalg.lstsq(cxx + alpha * np.eye(len(cxx)), cxy, rcond=None)[0]
        return cls(coef, float(mean_y - mean_x @ coef), alpha)


def fit_ols(codes: Array, targets: Array) -> RidgeHead:
    """Ordinary least squares head (alpha = 0)."""
    stats = RidgeStatistics(codes.shape[1])
    stats.update(codes, targets)
    return RidgeHead.from_statistics(stats, 0.0)


def fit_ridge(
    codes: Array,
    targets: Array,
    holdout: NDArray[np.bool_],
    alphas: Iterable[float] = RIDGE_ALPHAS,
) -> RidgeHead:
    """Selects alpha by RMSE on the `holdout` rows, then refits on all rows."""
    fit_stats, all_stats = (
        RidgeStatistics(codes.shape[1]),
        RidgeStatistics(codes.shape[1]),
    )
    fit_stats.update(codes[~holdout], targets[~holdout])
    all_stats.update(codes, targets)
    best_alpha = min(
        alphas,
        key=lambda alpha: rmse(
            targets[holdout],
            RidgeHead.from_statistics(fit_stats, alpha).predict(codes[holdout]),
        ),
    )
    return RidgeHead.from_statistics(all_stats, best_alpha)


# --- metrics ---


def rmse(a: Array, b: Array) -> float:
    """Sqrt of the mean of ‖a_i - b_i‖² over rows."""
    sq = (np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2
    if sq.ndim == 2:
        sq = sq.sum(1)
    return float(np.sqrt(sq.mean()))


class CompletenessAccumulator:
    """Completeness metrics, accumulated over chunks.

    RE, FE, MCE, L_g·RE, ATE, the relative L1 reconstruction error of h, L0 and the
    dead fraction. It also keeps f_A of every row (in update order), e.g. for AUROC.
    """

    def __init__(
        self,
        decoder_weight: Array,
        decoder_bias: Array,
        head: LinearHead,
        eta: Regressor,
        extra_etas: dict[str, Regressor] | None = None,
    ) -> None:
        self.decoder_weight = decoder_weight
        self.decoder_bias = decoder_bias
        self.head = head
        self.eta = eta
        self.extra_etas = extra_etas or {}
        self.baseline = baseline(decoder_bias, head)
        self.sse = dict.fromkeys(
            ("RE", "FE", "MCE", "ATE", *(f"MCE_{name}" for name in self.extra_etas)),
            0.0,
        )
        self.abs_residual = 0.0  # Σ |z - ẑ| over all entries
        self.abs_z = 0.0  # Σ |z| over all entries
        self.n = 0
        self.active = 0
        self.ever_active = np.zeros(len(decoder_weight), dtype=bool)
        self.f_a: list[Array] = []

    def update(self, f: Array, z: Array, codes: Array) -> None:
        f, z, codes = (np.asarray(a, np.float64) for a in (f, z, codes))
        z_hat = decode(codes, self.decoder_weight, self.decoder_bias)
        f_a = self.head(z_hat)
        phi = contributions(codes, self.decoder_weight, self.head)

        self.sse["RE"] += float(((z - z_hat) ** 2).sum())
        self.abs_residual += float(np.abs(z - z_hat).sum())
        self.abs_z += float(np.abs(z).sum())
        self.sse["FE"] += float(((f - f_a) ** 2).sum())
        self.sse["MCE"] += float(((f - self.eta.predict(codes)) ** 2).sum())
        for name, eta in self.extra_etas.items():
            self.sse[f"MCE_{name}"] += float(((f - eta.predict(codes)) ** 2).sum())
        self.sse["ATE"] += float(((f - (self.baseline + phi.sum(1))) ** 2).sum())
        self.n += len(f)
        nonzero = codes != 0
        self.active += int(nonzero.sum())
        self.ever_active |= nonzero.any(0)
        self.f_a.append(f_a)

    def compute(self) -> dict[str, float]:
        metrics = {key: float(np.sqrt(sse / self.n)) for key, sse in self.sse.items()}
        metrics["L_g_RE"] = self.head.lipschitz * metrics["RE"]
        # Σ_i Σ_d |z_id - ẑ_id| / Σ_i Σ_d |z_id| over all nuclei i and dimensions d
        metrics["relative_l1_error_h"] = self.abs_residual / self.abs_z
        metrics["L0"] = self.active / self.n
        metrics["dead_fraction"] = float((~self.ever_active).mean())
        return metrics

    @property
    def reconstructed(self) -> Array:
        return np.concatenate(self.f_a)


def completeness_metrics(
    f: Array,
    z: Array,
    codes: Array,
    decoder_weight: Array,
    decoder_bias: Array,
    head: LinearHead,
    eta: Regressor,
) -> dict[str, float]:
    """RE, FE, MCE, L_g·RE, ATE, L0 and dead fraction on one evaluation set."""
    accumulator = CompletenessAccumulator(decoder_weight, decoder_bias, head, eta)
    accumulator.update(f, z, codes)
    return accumulator.compute()


def logit_agreement(f: Array, f_a: Array) -> dict[str, float]:
    """How f_A tracks f: correlations and the shrinkage of f_A relative to f.

    `slope` and `intercept` regress f on f_A (f ≈ intercept + slope · f_A); a slope
    above 1 together with std(f_A) / std(f) below 1 means f_A is shrunk towards its
    mean.
    """
    f, f_a = np.asarray(f, np.float64), np.asarray(f_a, np.float64)
    cov = np.cov(f, f_a, bias=True)
    slope = cov[0, 1] / cov[1, 1]
    return {
        "pearson_f_fA": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
        "spearman_f_fA": float(spearmanr(f, f_a).statistic),
        "shrinkage_slope": float(slope),
        "shrinkage_intercept": float(f.mean() - slope * f_a.mean()),
        "std_ratio_fA_f": float(np.sqrt(cov[1, 1] / cov[0, 0])),
    }


def classification_metrics(labels: Array, scores: Array) -> dict[str, float]:
    """AUROC and AUPRC (average precision) of logits against binary labels."""
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
    }
