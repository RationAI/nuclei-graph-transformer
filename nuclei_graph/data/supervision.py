from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree
from torch import Tensor
from tqdm import tqdm


type Coords = NDArray[np.float32]

SOURCE_MAP = {
    "annotation": {"annot_df": "annot_uri"},
    "cam": {"cam_df": "cam_uri"},
    "agreement": {"annot_df": "annot_uri", "cam_df": "cam_uri"},
    "positive-agreement": {"annot_df": "annot_uri", "cam_df": "cam_uri"},
    "dense": {"dense_df": "dense_uri"},
}

COLUMN_MAP = {
    "annot_df": "annot_label",
    "cam_df": "cam_label",
    "dense_df": "pred_label",
}


class NucleiSupervision(ABC):
    def __init__(self, is_carcinoma: bool, balance_sampling: bool | None = False):
        self.is_carcinoma = is_carcinoma
        self.balance_sampling = balance_sampling

    @abstractmethod
    def get_targets(self, n: int) -> Tensor:
        """Returns nucleus-level targets.

        Args:
            n: Number of nuclei in the whole slide.

        Returns:
            Tensor of shape (n,): Target labels for each nucleus.
        """

    @abstractmethod
    def get_sup_mask(self, n: int) -> Tensor:
        """Returns nucleus-level supervision mask.

        Args:
            n: Number of nuclei in the whole slide.

        Returns:
            Boolean tensor of shape (n,): True for nuclei with confident labels.
        """

    @abstractmethod
    def get_seed_mask(self, n: int, centroids: Coords | None = None) -> Tensor:
        """Returns indices eligible as seeds for crop generation.

        Negative slides should return all True, while positive slides are expected to return
        confident positives or a balanced subset if `balance_sampling` is True.

        Args:
            n: Number of nuclei in the whole slide.
            centroids: (n, 2) array of nuclei centroids in micron units.

        Returns:
            Boolean tensor of shape (n,): True for nuclei eligible as seeds for crop generation.
        """

    @abstractmethod
    def get_positivity(self) -> float:
        """Returns the fraction of positive nuclei in [0.0, 1.0]."""

    def balance_seeds(
        self,
        pos_mask: Tensor,
        neg_mask: Tensor,
        centroids: Coords,
        inner_radius: float = 30.0,
        outer_radius: float = 200.0,
    ) -> Tensor:
        """Prioritizes negative seeds that are close to positive clusters.

        Args:
            pos_mask: Boolean mask of positive nuclei.
            neg_mask: Boolean mask of negative nuclei.
            centroids: (N, 2) array of nucleus centroids in micron units.
            inner_radius: Distance (in microns) within which negatives are most likely to be selected.
            outer_radius: Distance (in microns) beyond which negatives are unlikely to be selected.

        Returns:
            Boolean mask of selected seeds, including all positives and a balanced subset of negatives.
        """
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())

        if n_pos == 0 or n_neg <= n_pos:
            return pos_mask | neg_mask

        pos_indices = torch.nonzero(pos_mask).squeeze(1).cpu().numpy()
        neg_indices = torch.nonzero(neg_mask).squeeze(1).cpu().numpy()

        pos_coords = centroids[pos_indices]
        neg_coords = centroids[neg_indices]

        # build KDTree for positives to find distances for all negatives
        tree = KDTree(pos_coords)
        dist, _ = tree.query(neg_coords, k=1, workers=-1)  # dist to nearest positive

        # set high weight for negatives in the boundary zone (Gaussian with mu=inner_radius and std=outer_radius/2)
        weights = np.exp(-((dist - inner_radius) ** 2) / (2 * (outer_radius / 2) ** 2))

        # ensure far-away negatives have a small probability of being selected
        weights = weights + 1e-5

        # normalize to get a probability distribution
        weights /= weights.sum()
        # get as many negatives as positives, sampled according to the weights
        selected_neg_sub_indices = np.random.choice(
            len(neg_indices), size=n_pos, replace=False, p=weights
        )
        selected_neg_indices = neg_indices[selected_neg_sub_indices]

        balanced_mask = pos_mask.clone()  # include all positives
        balanced_mask[torch.from_numpy(selected_neg_indices).to(pos_mask.device)] = True

        return balanced_mask


class AnnotationNucleiSupervision(NucleiSupervision):
    """Supervision based on rough pathologist annotations."""

    def __init__(
        self,
        is_carcinoma: bool,
        annot_labels: Tensor,
        balance_sampling: bool | None = True,
    ):
        super().__init__(is_carcinoma, balance_sampling)
        self.annot_labels = annot_labels
        self.balance_sampling = self.balance_sampling

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        return self.annot_labels

    def get_sup_mask(self, n: int) -> Tensor:
        return torch.full((n,), True, dtype=torch.bool)

    def get_seed_mask(self, n: int, centroids: Coords | None = None) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)

        assert centroids is not None
        if self.balance_sampling:
            pos_mask = self.annot_labels == 1
            neg_mask = self.annot_labels == 0
            return self.balance_seeds(pos_mask, neg_mask, centroids)

        return self.annot_labels == 1

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        return float((self.annot_labels == 1).sum() / len(self.annot_labels))


class CAMNucleiSupervision(NucleiSupervision):
    """Supervision based on CAM labels only.

    Regions above a certain CAM threshold are considered positive, those below a certain
    threshold are negative, and those in between are ignored.
    """

    def __init__(
        self,
        is_carcinoma: bool,
        cam_labels: Tensor,
        balance_sampling: bool | None = True,
    ):
        super().__init__(is_carcinoma, balance_sampling)
        self.cam_labels = cam_labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        return self.cam_labels

    def get_sup_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), True, dtype=torch.bool)
        return self.cam_labels != -1

    def get_seed_mask(self, n: int, centroids: Coords | None = None) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)

        assert centroids is not None
        if self.balance_sampling:
            pos_mask = self.cam_labels == 1
            neg_mask = self.cam_labels == 0
            return self.balance_seeds(pos_mask, neg_mask, centroids)

        return self.cam_labels != -1

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        return float((self.cam_labels == 1).sum() / len(self.cam_labels))


class AgreementNucleiSupervision(NucleiSupervision):
    """Supervision based on the consensus between Annotations and CAM labels.

    The supervision mask is only valid where the annotation exactly matches the CAM label.
    """

    def __init__(
        self,
        is_carcinoma: bool,
        cam_labels: Tensor,
        annot_labels: Tensor,
        balance_sampling: bool | None = True,
    ):
        super().__init__(is_carcinoma, balance_sampling)
        self.cam_labels = cam_labels
        self.annot_labels = annot_labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        return self.annot_labels

    def get_sup_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), True, dtype=torch.bool)
        return self.annot_labels == self.cam_labels

    def get_seed_mask(self, n: int, centroids: Coords | None = None) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)

        assert centroids is not None
        if self.balance_sampling:
            pos_mask = (self.annot_labels == 1) & (self.cam_labels == 1)
            neg_mask = (self.annot_labels == 0) & (self.cam_labels == 0)
            return self.balance_seeds(pos_mask, neg_mask, centroids)

        return self.annot_labels == self.cam_labels

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        positive_sum = ((self.annot_labels == 1) & (self.cam_labels == 1)).sum()
        return float(positive_sum / len(self.annot_labels))


class PositiveAgreementNucleiSupervision(NucleiSupervision):
    """Strict supervision requiring both Annotation and CAM to agree on a positive label.

    Unlike Agreement, negatives in positive slides (0 == 0) are masked out and not supervised.
    """

    def __init__(
        self,
        is_carcinoma: bool,
        cam_labels: Tensor,
        annot_labels: Tensor,
    ):
        super().__init__(is_carcinoma)
        self.cam_labels = cam_labels
        self.annot_labels = annot_labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        return self.annot_labels

    def get_sup_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), True, dtype=torch.bool)
        return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_seed_mask(self, n: int, centroids: Coords | None = None) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        positive_sum = ((self.annot_labels == 1) & (self.cam_labels == 1)).sum()
        return float(positive_sum / len(self.annot_labels))


class DenseNucleiSupervision(NucleiSupervision):
    """Supervision where all provided nuclei are confidently labeled.

    Intended for dense nucleus-level training or model prediction-based evaluation.
    """

    def __init__(
        self, is_carcinoma: bool, labels: Tensor, balance_sampling: bool | None = False
    ):
        super().__init__(is_carcinoma, balance_sampling)
        self.labels = labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        return self.labels

    def get_sup_mask(self, n: int) -> Tensor:
        return torch.full((n,), True, dtype=torch.bool)

    def get_seed_mask(self, n: int, centroids: Coords | None = None) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)

        if self.balance_sampling and centroids is not None:
            pos_mask = self.labels == 1
            neg_mask = self.labels == 0
            return self.balance_seeds(pos_mask, neg_mask, centroids)

        return torch.ones(n, dtype=torch.bool)

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        return float(self.labels.mean())


@dataclass(frozen=True)
class SlideSupervision:
    slide_label: int
    nuclei_supervision: NucleiSupervision


@dataclass(frozen=True)
class DatasetSupervision:
    supervision_map: dict[str, SlideSupervision]

    @property
    def positivity_map(self) -> dict[str, float]:
        return {
            slide_id: slide_sup.nuclei_supervision.get_positivity()
            for slide_id, slide_sup in self.supervision_map.items()
        }


class SupervisionStrategy:
    def __init__(
        self,
        mode: str,
        annot_uri: str | None = None,
        cam_uri: str | None = None,
        dense_uri: str | None = None,
        balance_sampling: bool | None = False,
    ):
        self.annot_uri = annot_uri
        self.cam_uri = cam_uri
        self.dense_uri = dense_uri

        self.mode = mode
        self.required_sources = SOURCE_MAP[mode]
        self.balance_sampling = balance_sampling

        self._modes = {
            "annotation": AnnotationNucleiSupervision,
            "cam": CAMNucleiSupervision,
            "agreement": AgreementNucleiSupervision,
            "positive-agreement": PositiveAgreementNucleiSupervision,
            "dense": DenseNucleiSupervision,
        }

        if mode not in self._modes:
            raise ValueError(f"Unknown supervision mode: {mode}")

    def create(
        self,
        is_carcinoma: bool,
        annot_labels: Tensor | None = None,
        cam_labels: Tensor | None = None,
        dense_labels: Tensor | None = None,
    ) -> NucleiSupervision:
        supervision = self._modes[self.mode]
        if self.mode == "annotation":
            return supervision(is_carcinoma, annot_labels)
        elif self.mode == "cam":
            return supervision(is_carcinoma, cam_labels, self.balance_sampling)
        elif self.mode == "agreement":
            return supervision(
                is_carcinoma, cam_labels, annot_labels, self.balance_sampling
            )
        elif self.mode == "positive-agreement":
            return supervision(is_carcinoma, cam_labels, annot_labels)
        else:  # dense supervision
            return supervision(is_carcinoma, dense_labels, self.balance_sampling)


def build_supervision(
    sup_strategy: SupervisionStrategy,
    label_map: dict[str, int],
    sup_dfs: dict[str, pd.DataFrame | None],
) -> DatasetSupervision:
    """Constructs Supervision objects for each slide based on the provided strategy and DataFrames with supervision data.

    Args:
        sup_strategy (SupervisionStrategy): Strategy specifying which nuclei supervision type to use.
        label_map (dict[str, int]): Mapping from slide IDs to slide-level labels (0 for negative, 1 for positive).
        sup_dfs (dict[str, pd.DataFrame | None]): Dictionary containing dataframes with supervision data.

    Returns:
        DatasetSupervision: Object containing a mapping from slide IDs to `SlideSupervision` instances.
    """
    assert any(df is not None for df in sup_dfs.values())
    sources = [df for df in sup_dfs.values() if df is not None]

    df = (
        reduce(
            lambda left, right: pd.merge(
                left, right, on=["slide_id", "id"], how="inner", validate="1:1"
            ),
            sources,
        )
        .sort_values(["slide_id", "id"])
        .groupby("slide_id")
    )

    sup_map = {}
    for slide_id, label in tqdm(label_map.items(), desc="Building Supervision"):
        if label == 0:
            nuclei_sup = sup_strategy.create(is_carcinoma=False)
        else:
            group = df.get_group(slide_id)

            def get_labels(df_key: str, group: pd.DataFrame) -> Tensor | None:
                col = COLUMN_MAP[df_key]
                return (
                    torch.tensor(group[col].values, dtype=torch.float32)
                    if col in group
                    else None
                )

            nuclei_sup = sup_strategy.create(
                is_carcinoma=True,
                annot_labels=get_labels("annot_df", group),
                cam_labels=get_labels("cam_df", group),
                dense_labels=get_labels("dense_df", group),
            )

        sup_map[slide_id] = SlideSupervision(
            slide_label=int(label), nuclei_supervision=nuclei_sup
        )

    return DatasetSupervision(supervision_map=sup_map)
