import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def cosine_similarity(
    a: NDArray[np.floating], b: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Cosine similarity between the rows of `a` (k, d) and of `b` (m, d)."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a.astype(np.float64) @ b.astype(np.float64).T


def matched_cosines(
    reference: NDArray[np.floating], other: NDArray[np.floating]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Hungarian matching of concept directions, maximizing the total (signed) cosine.

    Returns, for every concept of `reference`, its matched concept in `other` and
    their cosine similarity.
    """
    similarity = cosine_similarity(reference, other)
    rows, cols = linear_sum_assignment(-similarity)
    order = np.argsort(rows)
    return cols[order], similarity[rows[order], cols[order]]
