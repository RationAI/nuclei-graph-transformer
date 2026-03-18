from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm


type Coords = NDArray[np.float32]

SOURCE_MAP = {
    "annotation": {"annot_df": "annot_uri"},
    "cam": {"cam_df": "cam_uri"},
    "agreement": {"annot_df": "annot_uri", "cam_df": "cam_uri"},
    "dense": {"dense_df": "dense_uri"},
}

COLUMN_MAP = {
    "annot_df": "annot_label",
    "cam_df": "cam_label",
    "dense_df": "pred_label",
}


class NucleiSupervision(ABC):
    def __init__(self, is_carcinoma: bool):
        self.is_carcinoma = is_carcinoma

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
    def get_positivity(self) -> float:
        """Returns the fraction of confident positive nuclei relative to total supervised nuclei."""

    def get_pos_seeds(self, n: int) -> list[int]:
        """Returns a list of indices for confident positive nuclei."""
        if not self.is_carcinoma:
            return []
        mask = (self.get_targets(n) == 1) & self.get_sup_mask(n)
        return torch.nonzero(mask).flatten().tolist()

    def get_neg_seeds(self, n: int) -> list[int]:
        """Returns a list of indices for confident negative nuclei."""
        mask = (self.get_targets(n) == 0) & self.get_sup_mask(n)
        return torch.nonzero(mask).flatten().tolist()


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
        return torch.full((n,), True, dtype=torch.bool)

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        n_sup = self.get_sup_mask(len(self.annot_labels)).sum().item()
        return float((self.annot_labels == 1).sum() / n_sup)


class CAMNucleiSupervision(NucleiSupervision):
    """Supervision based on CAM labels only.

    Regions above a certain CAM threshold are considered positive, those below a certain
    threshold are negative, and those in between are ignored.
    """

    def __init__(self, is_carcinoma: bool, cam_labels: Tensor):
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

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        n_sup = self.get_sup_mask(len(self.cam_labels)).sum().item()
        return float((self.cam_labels == 1).sum() / n_sup)


class AgreementNucleiSupervision(NucleiSupervision):
    """Supervision based on the consensus between Annotations and CAM labels.

    The supervision mask is only valid where the annotation exactly matches the CAM label.
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
        return self.annot_labels == self.cam_labels

    def get_positivity(self) -> float:
        if not self.is_carcinoma:
            return 0.0
        positive_sum = ((self.annot_labels == 1) & (self.cam_labels == 1)).sum()
        n_sup = self.get_sup_mask(len(self.annot_labels)).sum().item()
        return float(positive_sum / n_sup)


class DenseNucleiSupervision(NucleiSupervision):
    """Supervision where all provided nuclei are confidently labeled.

    Intended for dense nucleus-level training or model prediction-based evaluation.
    """

    def __init__(self, is_carcinoma: bool, labels: Tensor):
        super().__init__(is_carcinoma)
        self.labels = labels

    def get_targets(self, n: int) -> Tensor:
        if not self.is_carcinoma:
            return torch.full((n,), 0.0, dtype=torch.float32)
        return self.labels

    def get_sup_mask(self, n: int) -> Tensor:
        return torch.full((n,), True, dtype=torch.bool)

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
    ):
        self.annot_uri = annot_uri
        self.cam_uri = cam_uri
        self.dense_uri = dense_uri

        self.mode = mode
        self.required_sources = SOURCE_MAP[mode]

        self._modes = {
            "annotation": AnnotationNucleiSupervision,
            "cam": CAMNucleiSupervision,
            "agreement": AgreementNucleiSupervision,
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
            return supervision(is_carcinoma, cam_labels)
        elif self.mode == "agreement":
            return supervision(is_carcinoma, cam_labels, annot_labels)
        else:  # dense supervision
            return supervision(is_carcinoma, dense_labels)


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
    for slide_id, label in tqdm(label_map.items(), desc="Building Supervision"):
        if label == 0:
            nuclei_sup = sup_strategy.create(is_carcinoma=False)
        else:
            group = sup_groups.get_group(slide_id)

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
