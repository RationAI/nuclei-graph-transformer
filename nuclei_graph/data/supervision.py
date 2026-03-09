from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree
from torch import Tensor
from tqdm import tqdm


Coords = NDArray[np.float32]


class NucleiSupervision(ABC):
    def __init__(self, is_carcinoma: bool, balance_sampling: bool | None = None):
        """Abstract base class for nucleus-level supervision.

        The logic in subclasses only differs for the positive slides. For negative slides,
        all nuclei are implicitly assigned a target label of 0.0 and are fully supervised.
        """
        self.is_carcinoma = is_carcinoma
        self.balance_sampling = balance_sampling

    @abstractmethod
    def get_targets(self, n: int) -> Tensor:
        """Returns nucleus-level targets."""

    @abstractmethod
    def get_sup_mask(self, n: int) -> Tensor:
        """Returns supervision mask (True for confident labels)."""

    @abstractmethod
    def get_seed_mask(self, n: int, centroids: Coords | None = None) -> Tensor:
        """Returns indices eligible as seeds for crop generation.

        Negative slides should return all True, while positive slides
        are expected to return confident positives or a balanced subset
        if `balance_sampling` is True.

        Centroids are expected to be in the micron units as they are used
        for distance-based balancing of seeds.
        """

    @abstractmethod
    def get_positivity(self) -> float:
        """Returns the fraction of positive nuclei [0.0, 1.0]."""

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
        dist, _ = tree.query(neg_coords, k=1)  # distance to the nearest positive

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

    def __init__(self, is_carcinoma: bool, annot_labels: Tensor):
        super().__init__(is_carcinoma)
        self.annot_labels = annot_labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        return self.annot_labels

    def get_sup_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), True, dtype=torch.bool)
        return self.annot_labels == 1

    def get_seed_mask(self, n: int, centroids: Coords | None = None) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        return self.annot_labels == 1

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        return float((self.annot_labels == 1).sum() / len(self.annot_labels))


class CAMNucleiSupervision(NucleiSupervision):
    """Supervision based on CAM labels.

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


class PredictionNucleiSupervision(NucleiSupervision):
    """Supervision based on model predictions (e.g., Virchow2).

    This mode is intended only for testing and prediction evaluation.
    All nuclei are supervised based on the provided prediction labels.
    """

    def __init__(self, is_carcinoma: bool, pred_labels: Tensor):
        super().__init__(is_carcinoma)
        self.pred_labels = pred_labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        return self.pred_labels

    def get_sup_mask(self, n: int) -> Tensor:
        return torch.full((n,), True, dtype=torch.bool)

    def get_seed_mask(self, n: int, centroids: Coords | None = None) -> Tensor:
        return torch.full((n,), True, dtype=torch.bool)  # dummy output

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        return float(self.pred_labels.mean())


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
        pred_uri: str | None = None,
        balance_sampling: bool | None = None,
    ):
        self.annot_uri = annot_uri
        self.cam_uri = cam_uri
        self.pred_uri = pred_uri
        self.mode = mode
        self.balance_sampling = balance_sampling

        self._modes = {
            "annotation": AnnotationNucleiSupervision,
            "cam": CAMNucleiSupervision,
            "agreement": AgreementNucleiSupervision,
            "positive-agreement": PositiveAgreementNucleiSupervision,
            "prediction": PredictionNucleiSupervision,
        }
        if mode not in self._modes:
            raise ValueError(f"Unknown supervision mode: {mode}")

    def create(
        self,
        is_carcinoma: bool,
        annot_labels: Tensor | None = None,
        cam_labels: Tensor | None = None,
        pred_labels: Tensor | None = None,
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
        else:  # prediction
            return supervision(is_carcinoma, pred_labels)


def build_supervision(
    sup_strategy: SupervisionStrategy,
    label_map: dict[str, int],
    df_annot: pd.DataFrame | None = None,
    df_cam: pd.DataFrame | None = None,
    df_pred: pd.DataFrame | None = None,
) -> DatasetSupervision:
    assert not (df_annot is None and df_cam is None and df_pred is None)

    sources = [df for df in [df_annot, df_cam, df_pred] if df is not None]

    df_merged = sources[0]
    for next_source in sources[1:]:
        df_merged = pd.merge(
            df_merged, next_source, on=["slide_id", "id"], how="inner", validate="1:1"
        )

    df_merged = df_merged.sort_values(["slide_id", "id"])
    grouped = df_merged.groupby("slide_id")

    sup_map = {}
    for slide_id, label in tqdm(label_map.items(), desc="Building Supervision"):
        if label == 0:
            nuclei_sup = sup_strategy.create(is_carcinoma=False)
        else:
            group = grouped.get_group(slide_id)

            def get_col(name, g=group):
                if name in g.columns:
                    return torch.tensor(g[name].values, dtype=torch.float32)
                return None

            nuclei_sup = sup_strategy.create(
                is_carcinoma=True,
                annot_labels=get_col("annot_label"),
                cam_labels=get_col("cam_label"),
                pred_labels=get_col("pred_label"),
            )
        sup_map[slide_id] = SlideSupervision(
            slide_label=int(label), nuclei_supervision=nuclei_sup
        )

    return DatasetSupervision(supervision_map=sup_map)
