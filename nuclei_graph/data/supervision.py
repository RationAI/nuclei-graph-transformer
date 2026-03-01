from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm


class NucleiSupervision(ABC):
    def __init__(self, is_carcinoma: bool):
        """Abstract base class for nucleus-level supervision.

        The logic in subclasses only differs for the positive slides. For negative slides,
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
        """Returns indices eligible as seeds for crop generation.

        Negative slides should return all True, while positive slides
        are expected to return confident positives.
        """

    @abstractmethod
    def get_positivity(self) -> float:
        """Returns the fraction of positive nuclei [0.0, 1.0]."""


class AnnotationNucleiSupervision(NucleiSupervision):
    """Supervision based on rough pathologist annotations."""

    def __init__(self, is_carcinoma: bool, annot_labels: Tensor | None = None):
        super().__init__(is_carcinoma)
        self.annot_labels = annot_labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        assert self.annot_labels is not None
        return self.annot_labels

    def get_sup_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), True, dtype=torch.bool)
        assert self.annot_labels is not None
        return self.annot_labels == 1

    def get_seed_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        assert self.annot_labels is not None
        return self.annot_labels == 1

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        assert self.annot_labels is not None
        return float((self.annot_labels == 1).sum() / len(self.annot_labels))


class CAMNucleiSupervision(NucleiSupervision):
    """Supervision based on CAM labels.

    Regions above a certain CAM threshold are considered positive, those below a certain
    threshold are negative, and those in between are ignored.
    The thresholds are determined by the `cam_thr_type` parameter in the `SupervisionStrategy` class.
    """

    def __init__(self, is_carcinoma: bool, cam_labels: Tensor | None = None):
        super().__init__(is_carcinoma)
        self.cam_labels = cam_labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        assert self.cam_labels is not None
        return self.cam_labels

    def get_sup_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), True, dtype=torch.bool)
        assert self.cam_labels is not None
        return self.cam_labels != -1

    def get_seed_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        assert self.cam_labels is not None
        return self.cam_labels == 1

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        assert self.cam_labels is not None
        return float((self.cam_labels == 1).sum() / len(self.cam_labels))


class AgreementNucleiSupervision(NucleiSupervision):
    """Supervision based on the consensus between Annotations and CAM labels.

    The supervision mask is only valid where the annotation exactly matches the CAM label.
    """

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
        assert self.annot_labels is not None
        return self.annot_labels

    def get_sup_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), True, dtype=torch.bool)
        assert self.annot_labels is not None and self.cam_labels is not None
        return self.annot_labels == self.cam_labels

    def get_seed_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        assert self.annot_labels is not None and self.cam_labels is not None
        return self.annot_labels == self.cam_labels
        # return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        assert self.annot_labels is not None and self.cam_labels is not None
        return float(
            ((self.annot_labels == 1) & (self.cam_labels == 1)).sum()
            / len(self.annot_labels)
        )


class PositiveAgreementNucleiSupervision(NucleiSupervision):
    """Strict supervision requiring both Annotation and CAM to agree on a positive label.

    Unlike Agreement, negatives in positive slides (0 == 0) are masked out and not supervised.
    """

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
        assert self.annot_labels is not None
        return self.annot_labels

    def get_sup_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), True, dtype=torch.bool)
        assert self.annot_labels is not None and self.cam_labels is not None
        return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_seed_mask(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        assert self.annot_labels is not None and self.cam_labels is not None
        return (self.annot_labels == 1) & (self.cam_labels == 1)

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        assert self.annot_labels is not None and self.cam_labels is not None
        return float(
            ((self.annot_labels == 1) & (self.cam_labels == 1)).sum()
            / len(self.annot_labels)
        )


@dataclass(frozen=True)
class SlideSupervision:
    slide_label: int
    nuclei_supervision: NucleiSupervision


@dataclass(frozen=True)
class DatasetSupervision:
    supervision_map: dict[str, SlideSupervision]


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
        if self.mode == "annotation":
            return supervision(is_carcinoma, annot_labels=annot_labels)
        elif self.mode == "cam":
            return supervision(is_carcinoma, cam_labels=cam_labels)
        else:  # agreement modes
            return supervision(
                is_carcinoma, cam_labels=cam_labels, annot_labels=annot_labels
            )


def build_supervision(
    sup_strategy: SupervisionStrategy,
    df_annot: pd.DataFrame,
    df_cam: pd.DataFrame | None,
    label_map: dict[str, int],
) -> DatasetSupervision:

    df_merged = (
        pd.merge(df_annot, df_cam, on=["slide_id", "id"], how="inner", validate="1:1")
        if df_cam is not None
        else df_annot
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
            cam = (
                torch.tensor(group["cam_label"].values, dtype=torch.float32)
                if df_cam is not None
                else None
            )
            nuclei_sup = sup_strategy.create(
                is_carcinoma=True,
                annot_labels=annot,
                cam_labels=cam,
            )

        sup_map[slide_id] = SlideSupervision(
            slide_label=int(label), nuclei_supervision=nuclei_sup
        )

    return DatasetSupervision(supervision_map=sup_map)
