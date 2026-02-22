from abc import ABC, abstractmethod

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from nuclei_graph.nuclei_graph_typing import DatasetSupervision, SlideSupervision


class NucleiSupervision(ABC):
    def __init__(self, n: int, is_carcinoma: bool):
        self.n = n
        self.is_carcinoma = is_carcinoma
        self.targets_negative = torch.full((n,), 0.0, dtype=torch.float32)
        self.sup_mask_all = torch.full((n,), True, dtype=torch.bool)
        self.seed_mask_all = torch.arange(n)

    @abstractmethod
    def get_targets(self) -> Tensor:
        """Returns nucleus-level targets."""

    @abstractmethod
    def get_sup_mask(self) -> Tensor:
        """Returns supervision mask (True for confident labels)."""

    @abstractmethod
    def get_seed_mask(self) -> Tensor:
        """Returns indices eligible as seeds for crop generation."""

    @abstractmethod
    def get_positivity(self) -> float:
        """Returns the fraction of positive nuclei [0.0, 1.0]."""


class AnnotationNucleiSupervision(NucleiSupervision):
    """Supervision based on annotations."""

    def __init__(self, n: int, is_carcinoma: bool, annot_labels: Tensor):
        super().__init__(n, is_carcinoma)
        self.annot_labels = annot_labels

    def get_targets(self) -> Tensor:
        if not self.is_carcinoma:
            return self.targets_negative
        return self.annot_labels

    def get_sup_mask(self) -> Tensor:
        if not self.is_carcinoma:
            return self.sup_mask_all
        return self.annot_labels == 1

    def get_seed_mask(self) -> Tensor:
        if not self.is_carcinoma:
            return self.seed_mask_all
        return self.annot_labels == 1

    def get_positivity(self) -> float:
        if not self.is_carcinoma or self.n == 0:
            return 0.0
        return float((self.annot_labels == 1).sum() / self.n)


class CAMNucleiSupervision(NucleiSupervision):
    """Supervision based on CAM only."""

    def __init__(self, n: int, is_carcinoma: bool, cam_labels: Tensor):
        super().__init__(n, is_carcinoma)
        self.cam_labels = cam_labels

    def get_targets(self) -> Tensor:
        if not self.is_carcinoma:
            return self.targets_negative
        return self.cam_labels

    def get_sup_mask(self) -> Tensor:
        if not self.is_carcinoma:
            return self.sup_mask_all
        return self.cam_labels != -1

    def get_seed_mask(self) -> Tensor:
        if not self.is_carcinoma:
            return self.seed_mask_all
        return self.cam_labels == 1

    def get_positivity(self) -> float:
        if not self.is_carcinoma or self.n == 0:
            return 0.0
        return float((self.cam_labels == 1).sum() / self.n)


class AgreementNucleiSupervision(NucleiSupervision):
    """Supervision where annotation and CAM agree."""

    def __init__(
        self, n: int, is_carcinoma: bool, cam_labels: Tensor, annot_labels: Tensor
    ):
        super().__init__(n, is_carcinoma)
        self.cam_labels = cam_labels
        self.annot_labels = annot_labels

    def get_targets(self) -> Tensor:
        if not self.is_carcinoma:
            return self.targets_negative
        return self.annot_labels

    def get_sup_mask(self) -> Tensor:
        if not self.is_carcinoma:
            return self.sup_mask_all
        return self.annot_labels == self.cam_labels

    def get_seed_mask(self) -> Tensor:
        if not self.is_carcinoma:
            return self.seed_mask_all
        return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_positivity(self) -> float:
        if not self.is_carcinoma or self.n == 0:
            return 0.0
        return float(((self.annot_labels == 1) & (self.cam_labels == 1)).sum() / self.n)


class PositiveAgreementNucleiSupervision(NucleiSupervision):
    """Supervision only for nuclei that are positive in both annotation and CAM."""

    def __init__(
        self, n: int, is_carcinoma: bool, cam_labels: Tensor, annot_labels: Tensor
    ):
        super().__init__(n, is_carcinoma)
        self.cam_labels = cam_labels
        self.annot_labels = annot_labels

    def get_targets(self) -> Tensor:
        if not self.is_carcinoma:
            return self.targets_negative
        return self.annot_labels

    def get_sup_mask(self) -> Tensor:
        if not self.is_carcinoma:
            return self.sup_mask_all
        return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_seed_mask(self) -> Tensor:
        if not self.is_carcinoma:
            return self.seed_mask_all
        return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_positivity(self) -> float:
        if not self.is_carcinoma or self.n == 0:
            return 0.0
        return float(((self.annot_labels == 1) & (self.cam_labels == 1)).sum() / self.n)


class SupervisionStrategy:
    def __init__(self, mode: str, cam_thr_type: str | None = None):
        self.cam_thr_type = cam_thr_type
        self.mode = mode
        self._modes = {
            "annotation": AnnotationNucleiSupervision,
            "cam": CAMNucleiSupervision,
            "agreement": AgreementNucleiSupervision,
            "positive-agreement": PositiveAgreementNucleiSupervision,
        }
        if mode not in self._modes:
            raise ValueError(f"Unknown supervision mode: {mode}")

    def create(
        self,
        n: int,
        is_carcinoma: bool,
        annot_labels: Tensor,
        cam_labels: Tensor,
    ) -> NucleiSupervision:
        supervision = self._modes[self.mode]

        if self.mode == "annotation":
            assert annot_labels is not None
            return supervision(n, is_carcinoma, annot_labels)
        elif self.mode == "cam":
            assert cam_labels is not None
            return supervision(n, is_carcinoma, cam_labels)
        else:  # agreement modes
            assert annot_labels is not None and cam_labels is not None
            return supervision(n, is_carcinoma, cam_labels, annot_labels)


def build_supervision(
    sup_strategy: SupervisionStrategy,
    df_annot: pd.DataFrame,
    df_cam: pd.DataFrame,
    label_map: dict[str, int],
) -> DatasetSupervision:
    """Builds a DatasetSupervision dataclass.

    Args:
        sup_strategy: An instance of SupervisionStrategy defining the type of supervision to use.
        df_annot: DataFrame containing annotation labels with columns "slide_id" (str), "id" (str), and "annot_label" (int).
        df_cam: DataFrame containing CAM labels with columns "slide_id" (str), "id" (str), and "cam_label" (int).
        label_map: Mapping from slide_id to slide-level label (int).
    """
    df_merged = pd.merge(
        df_annot, df_cam, on=["slide_id", "id"], how="inner", validate="1:1"
    )

    df_merged = df_merged.sort_values(["slide_id", "id"])
    grouped = df_merged.groupby("slide_id")

    sup_map = {}
    for slide_id, label in tqdm(label_map.items(), desc="Building Supervision"):
        group = grouped.get_group(slide_id)

        n = len(group)
        annot = torch.tensor(group["annot_label"].values, dtype=torch.float32)
        cam = torch.tensor(group["cam_label"].values, dtype=torch.float32)

        nuclei_sup = sup_strategy.create(
            n=n,
            is_carcinoma=bool(label),
            annot_labels=annot,
            cam_labels=cam,
        )

        sup_map[slide_id] = SlideSupervision(
            slide_label=int(label), nuclei_supervision=nuclei_sup
        )

    return DatasetSupervision(supervision_map=sup_map)
