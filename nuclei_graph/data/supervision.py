from abc import ABC, abstractmethod

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from nuclei_graph.nuclei_graph_typing import DatasetSupervision, SlideSupervision


class NucleiSupervision(ABC):
    def __init__(self, is_carcinoma: bool):
        """Abstract base class for nucleus-level supervision.

        The logic in subclasses only applies to positive slides. For negative slides,
        all nuclei are implicitly assigned a target label of 0.0 and are fully supervised.
        """
        self.is_carcinoma = is_carcinoma

    @abstractmethod
    def get_targets(self, n: int) -> Tensor:
        """Returns nucleus-level targets."""

    @abstractmethod
    def get_sup_mask(self, n: int) -> Tensor:
        """Returns supervision mask (True for confident labels)."""

    @abstractmethod
    def get_seed_mask(self, n: int) -> Tensor:
        """Returns indices eligible as seeds for crop generation."""

    @abstractmethod
    def get_positivity(self) -> float:
        """Returns the fraction of positive nuclei [0.0, 1.0]."""


class AnnotationNucleiSupervision(NucleiSupervision):
    def __init__(self, is_carcinoma: bool, annot_labels: Tensor | None = None):
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

    def get_seed_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        return self.annot_labels == 1

    def get_positivity(self) -> float:
        if (
            not self.is_carcinoma
            or self.annot_labels is None
            or len(self.annot_labels) == 0
        ):
            return 0.0
        return float((self.annot_labels == 1).sum() / len(self.annot_labels))


class CAMNucleiSupervision(NucleiSupervision):
    def __init__(self, is_carcinoma: bool, cam_labels: Tensor | None = None):
        super().__init__(is_carcinoma)
        self.cam_labels = cam_labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        return self.cam_labels

    def get_sup_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), True, dtype=torch.bool)
        return self.cam_labels != -1

    def get_seed_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        return self.cam_labels == 1

    def get_positivity(self) -> float:
        if (
            not self.is_carcinoma
            or self.cam_labels is None
            or len(self.cam_labels) == 0
        ):
            return 0.0
        return float((self.cam_labels == 1).sum() / len(self.cam_labels))


class AgreementNucleiSupervision(NucleiSupervision):
    def __init__(
        self,
        is_carcinoma: bool,
        cam_labels: Tensor | None = None,
        annot_labels: Tensor | None = None,
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
        return self.annot_labels == self.cam_labels

    def get_seed_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_positivity(self) -> float:
        if (
            not self.is_carcinoma
            or self.annot_labels is None
            or len(self.annot_labels) == 0
        ):
            return 0.0
        return float(
            ((self.annot_labels == 1) & (self.cam_labels == 1)).sum()
            / len(self.annot_labels)
        )


class PositiveAgreementNucleiSupervision(NucleiSupervision):
    def __init__(
        self,
        is_carcinoma: bool,
        cam_labels: Tensor | None = None,
        annot_labels: Tensor | None = None,
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

    def get_seed_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_positivity(self) -> float:
        if (
            not self.is_carcinoma
            or self.annot_labels is None
            or len(self.annot_labels) == 0
        ):
            return 0.0
        return float(
            ((self.annot_labels == 1) & (self.cam_labels == 1)).sum()
            / len(self.annot_labels)
        )


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
        is_carcinoma: bool,
        annot_labels: Tensor | None = None,
        cam_labels: Tensor | None = None,
    ) -> NucleiSupervision:
        supervision = self._modes[self.mode]
        return supervision(is_carcinoma, annot_labels, cam_labels)


def build_supervision(
    sup_strategy: SupervisionStrategy,
    df_annot: pd.DataFrame,
    df_cam: pd.DataFrame,
    label_map: dict[str, int],
) -> DatasetSupervision:

    df_merged = pd.merge(
        df_annot, df_cam, on=["slide_id", "id"], how="inner", validate="1:1"
    )
    df_merged = df_merged.sort_values(["slide_id", "id"])
    grouped = df_merged.groupby("slide_id")

    sup_map = {}
    for slide_id, label in tqdm(label_map.items(), desc="Building Supervision"):
        if label == 0:
            nuclei_sup = sup_strategy.create(is_carcinoma=False)
        else:
            group = grouped.get_group(slide_id)
            annot = torch.tensor(group["annot_label"].values, dtype=torch.float32)
            cam = torch.tensor(group["cam_label"].values, dtype=torch.float32)
            nuclei_sup = sup_strategy.create(
                is_carcinoma=True,
                annot_labels=annot,
                cam_labels=cam,
            )

        sup_map[slide_id] = SlideSupervision(
            slide_label=int(label), nuclei_supervision=nuclei_sup
        )

    return DatasetSupervision(supervision_map=sup_map)
