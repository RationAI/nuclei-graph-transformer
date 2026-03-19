from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import ClassVar

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm


COL_TO_KWARG = {
    "annot_label": "annot_labels",
    "cam_label": "cam_labels",
    "pred_label": "pred_labels",
}


class NucleiSupervision(ABC):
    def __init__(self, is_carcinoma: bool):
        self.is_carcinoma = is_carcinoma

    @abstractmethod
    def _get_sup_mask(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_targets(self) -> Tensor:
        raise NotImplementedError

    def get_sup_mask(self, n: int) -> Tensor:
        """Boolean mask of shape (n,) indicating confident nucleus-level labels."""
        if not self.is_carcinoma:
            return torch.ones(n, dtype=torch.bool)
        return self._get_sup_mask()

    def get_targets(self, n: int) -> Tensor:
        """Returns a tensor of shape (n,) with binary nucleus-level labels."""
        if not self.is_carcinoma:
            return torch.zeros(n, dtype=torch.float32)
        return self._get_targets()

    def get_neg_seeds(self, n: int) -> list[int]:
        """Returns a list of indices for confident negative nuclei."""
        mask = (self.get_targets(n) == 0) & self.get_sup_mask(n)
        return torch.nonzero(mask).flatten().tolist()

    def get_pos_seeds(self, n: int) -> list[int]:
        """Returns a list of indices for confident positive nuclei."""
        if not self.is_carcinoma:
            return []
        mask = (self.get_targets(n) == 1) & self.get_sup_mask(n)
        return torch.nonzero(mask).flatten().tolist()

    def get_positivity(self) -> float:
        """Fraction of confident positive nuclei relative to total supervised."""
        if not self.is_carcinoma:
            return 0.0

        targets = self._get_targets()
        mask = self._get_sup_mask()

        n_sup = mask.sum().item()
        if n_sup == 0:
            return 0.0

        return float((targets[mask] == 1).float().mean())

    def get_pos_count(self) -> int:
        """Returns the number of confident (supervised) positive nuclei."""
        if not self.is_carcinoma:
            return 0
        targets = self._get_targets()
        mask = self._get_sup_mask()
        return int(((targets == 1) & mask).sum().item())


class DenseNucleiSupervision(NucleiSupervision):
    """Supervision where all provided nuclei are confidently labeled."""

    def __init__(self, is_carcinoma: bool, labels: Tensor):
        super().__init__(is_carcinoma)
        self.labels = labels

    def _get_targets(self) -> Tensor:
        return self.labels

    def _get_sup_mask(self) -> Tensor:
        return torch.ones(len(self.labels), dtype=torch.bool)


class AnnotationNucleiSupervision(DenseNucleiSupervision):
    """Supervision based on rough pathologist annotations."""

    def __init__(self, is_carcinoma: bool, annot_labels: Tensor):
        super().__init__(is_carcinoma, labels=annot_labels)


class PredictionNucleiSupervision(DenseNucleiSupervision):
    """Supervision based on model predictions."""

    def __init__(self, is_carcinoma: bool, pred_labels: Tensor):
        super().__init__(is_carcinoma, labels=pred_labels)


class CAMNucleiSupervision(NucleiSupervision):
    """Supervision based on CAM labels only.

    Regions above a certain CAM threshold (1) are considered positive, those below
    a certain threshold (0) are negative, and those in between (-1) are ignored.
    """

    def __init__(self, is_carcinoma: bool, cam_labels: Tensor):
        super().__init__(is_carcinoma)
        self.cam_labels = cam_labels

    def _get_targets(self) -> Tensor:
        return self.cam_labels

    def _get_sup_mask(self) -> Tensor:
        return self.cam_labels != -1


class AgreementNucleiSupervision(NucleiSupervision):
    """Supervision based on the consensus between Annotations and CAM labels.

    The supervision mask is only valid where the annotation matches the CAM label.
    """

    def __init__(self, is_carcinoma: bool, cam_labels: Tensor, annot_labels: Tensor):
        super().__init__(is_carcinoma)
        self.cam_labels, self.annot_labels = cam_labels, annot_labels

    def _get_targets(self) -> Tensor:
        return self.annot_labels

    def _get_sup_mask(self) -> Tensor:
        return self.annot_labels == self.cam_labels


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

    @property
    def pos_count_map(self) -> dict[str, int]:
        return {
            slide_id: slide_sup.nuclei_supervision.get_pos_count()
            for slide_id, slide_sup in self.supervision_map.items()
        }


class SupervisionStrategy:
    STRATEGY_MAP: ClassVar = {
        "annotation": (AnnotationNucleiSupervision, ["annot_labels"]),
        "cam": (CAMNucleiSupervision, ["cam_labels"]),
        "agreement": (AgreementNucleiSupervision, ["annot_labels", "cam_labels"]),
        "prediction": (PredictionNucleiSupervision, ["pred_labels"]),
    }

    def __init__(self, mode: str, **uris):
        self.mode = mode
        self.uris = uris
        if mode not in self.STRATEGY_MAP:
            raise ValueError(f"Unknown mode: {mode}")

    def create(self, is_carcinoma: bool, **all_labels) -> NucleiSupervision:
        sup_class, required_keys = self.STRATEGY_MAP[self.mode]
        filtered_labels = {k: all_labels[k] for k in required_keys}

        if is_carcinoma:
            for k, v in filtered_labels.items():
                assert len(v) > 0, f"Missing required label {k} for slide"

        return sup_class(is_carcinoma, **filtered_labels)


def build_supervision(
    strategy: SupervisionStrategy,
    carcinoma_map: dict[str, bool],
    sup_dfs: dict[str, pd.DataFrame | None],
) -> DatasetSupervision:
    """Constructs Supervision for each slide based on the provided strategy and supervision data."""
    sources = [df for df in sup_dfs.values() if df is not None]
    assert sources

    sup_groups = (
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
    empty_t = torch.empty(0, dtype=torch.float32)

    for slide_id, is_carcinoma in tqdm(
        carcinoma_map.items(), desc="Building Supervision"
    ):
        # negative slides have no supervision data
        labels = dict.fromkeys(COL_TO_KWARG.values(), empty_t)

        if is_carcinoma:
            group = sup_groups.get_group(slide_id)
            for col, kwarg in COL_TO_KWARG.items():
                if col in group.columns:
                    labels[kwarg] = torch.from_numpy(group[col].values).float()

        nuclei_sup = strategy.create(is_carcinoma, **labels)
        sup_map[slide_id] = SlideSupervision(int(is_carcinoma), nuclei_sup)

    return DatasetSupervision(sup_map)
