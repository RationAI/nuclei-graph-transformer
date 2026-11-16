import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm


IGNORE_KEY = "ignore_dir"
# A label directory holding this file has 0 = "unlabelled" on positive slides (non-exhaustive
# annotation, e.g. the iCAIRD re-annotated test slides): only `annotation_sparse` may read it.
SPARSE_ONLY_MARKER = "_SPARSE_ONLY"


@dataclass(frozen=True)
class IdsDigest:
    """The count and SHA-256 of a sequence of nuclei ids, compared as `str`.

    Stands in for the ids themselves (64-character hashes: ~8 GB for the 131M iCAIRD
    training nuclei) in the check that labels belong to a slide's nuclei.
    """

    count: int
    sha256: str

    @classmethod
    def of(cls, ids: Iterable[object]) -> "IdsDigest":
        ids_str = [str(i) for i in ids]
        return cls(
            len(ids_str), hashlib.sha256("\0".join(ids_str).encode()).hexdigest()
        )

    def check(self, ids: NDArray[np.generic], what: str) -> None:
        """As `nuclei_features.check_ids`: the stored ids equal `ids`, the slide's."""
        assert self.count == len(ids), (
            f"{what}: {self.count} nuclei, the slide has {len(ids)}."
        )
        assert self == IdsDigest.of(ids.tolist()), (
            f"{what}: the nuclei ids differ from the slide's; "
            "regenerate from the current nuclei."
        )


class NucleiSupervision(ABC):
    """Base nuclei supervision.

    `ignore` optionally marks nuclei (e.g., "Another pathology" regions) that are
    excluded from supervision regardless of the slide label. `ids` and `ignore_ids`
    are digests of the nuclei ids (sorted) that the labels and `ignore` were read for;
    the datasets check them against the slide's nuclei with `check_ids`.
    """

    ids: IdsDigest | None = None
    ignore_ids: IdsDigest | None = None

    def __init__(self, is_carcinoma: bool, ignore: Tensor | None = None):
        self.is_carcinoma = is_carcinoma
        self.ignore = ignore

    def check_ids(self, ids: NDArray[np.generic]) -> None:
        """Asserts that the labels belong to `ids`, the slide's nuclei sorted by id."""
        if self.ids is not None:
            self.ids.check(ids, "the nuclei labels")
        if self.ignore_ids is not None:
            self.ignore_ids.check(ids, "the ignore labels")

    @abstractmethod
    def _get_sup_mask(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_targets(self) -> Tensor:
        raise NotImplementedError

    def get_sup_mask(self, n: int) -> Tensor:
        """Boolean mask of shape (n,) indicating confident nucleus-level labels."""
        mask = (
            self._get_sup_mask()
            if self.is_carcinoma
            else torch.ones(n, dtype=torch.bool)
        )
        if self.ignore is not None:
            assert len(self.ignore) == n, "Ignore mask does not match nuclei count"
            mask = mask & ~self.ignore
        return mask

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
        mask = self.get_sup_mask(len(targets))

        n_sup = mask.sum().item()
        if n_sup == 0:
            return 0.0

        return float((targets[mask] == 1).float().mean())

    def get_pos_count(self) -> int:
        """Returns the number of confident (supervised) positive nuclei."""
        if not self.is_carcinoma:
            return 0
        targets = self._get_targets()
        mask = self.get_sup_mask(len(targets))
        return int(((targets == 1) & mask).sum().item())


class DenseNucleiSupervision(NucleiSupervision):
    """Supervision where all provided nuclei are confidently labeled."""

    def __init__(
        self, is_carcinoma: bool, labels: Tensor, ignore: Tensor | None = None
    ):
        super().__init__(is_carcinoma, ignore)
        self.labels = labels

    def _get_targets(self) -> Tensor:
        return self.labels

    def _get_sup_mask(self) -> Tensor:
        return torch.ones(len(self.labels), dtype=torch.bool)


class AnnotationNucleiSupervision(DenseNucleiSupervision):
    """Supervision based on rough pathologist annotations."""

    def __init__(
        self, is_carcinoma: bool, annot_label: Tensor, ignore: Tensor | None = None
    ):
        super().__init__(is_carcinoma, labels=annot_label, ignore=ignore)


class SparseAnnotationNucleiSupervision(AnnotationNucleiSupervision):
    """Supervision based on non-exhaustive pathologist annotations.

    In positive slides, only nuclei inside annotations are supervised (as positive);
    nuclei outside annotations are unlabeled. Negatives come from negative slides only.
    """

    def _get_sup_mask(self) -> Tensor:
        return self.labels == 1


class PredictionNucleiSupervision(DenseNucleiSupervision):
    """Supervision based on model predictions."""

    def __init__(
        self, is_carcinoma: bool, pred_label: Tensor, ignore: Tensor | None = None
    ):
        super().__init__(is_carcinoma, labels=pred_label, ignore=ignore)


class CAMNucleiSupervision(NucleiSupervision):
    """Supervision based on CAM labels only.

    Regions above a certain CAM threshold (1) are considered positive, those below
    a certain threshold (0) are negative, and those in between (-1) are ignored.
    """

    def __init__(
        self, is_carcinoma: bool, cam_label: Tensor, ignore: Tensor | None = None
    ):
        super().__init__(is_carcinoma, ignore)
        self.cam_labels = cam_label

    def _get_targets(self) -> Tensor:
        return self.cam_labels

    def _get_sup_mask(self) -> Tensor:
        return self.cam_labels != -1


class AgreementNucleiSupervision(NucleiSupervision):
    """Supervision based on the consensus between Annotations and CAM labels.

    The supervision mask is only valid where the annotation matches the CAM label.
    """

    def __init__(
        self,
        is_carcinoma: bool,
        cam_label: Tensor,
        annot_label: Tensor,
        ignore: Tensor | None = None,
    ):
        super().__init__(is_carcinoma, ignore)
        self.cam_labels, self.annot_labels = cam_label, annot_label

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
        "annotation": (AnnotationNucleiSupervision, ["annot_label"]),
        "annotation_sparse": (SparseAnnotationNucleiSupervision, ["annot_label"]),
        "cam": (CAMNucleiSupervision, ["cam_label"]),
        "agreement": (AgreementNucleiSupervision, ["annot_label", "cam_label"]),
        "prediction": (PredictionNucleiSupervision, ["pred_label"]),
    }

    def __init__(self, mode: str, **paths: str | None):
        self.mode = mode
        self.paths = paths
        if mode not in self.STRATEGY_MAP:
            raise ValueError(f"Unknown mode: {mode}")
        annot_dir = paths.get("annot_dir")
        if (
            annot_dir is not None
            and (Path(annot_dir) / SPARSE_ONLY_MARKER).exists()
            and mode != "annotation_sparse"
        ):
            raise ValueError(
                f"{annot_dir} holds sparse labels (0 = unlabelled); use annotation_sparse, not {mode}"
            )

    def create(
        self,
        is_carcinoma: bool,
        ignore: Tensor | None = None,
        ids: IdsDigest | None = None,
        ignore_ids: IdsDigest | None = None,
        **all_labels: Tensor,
    ) -> NucleiSupervision:
        sup_class, required_keys = self.STRATEGY_MAP[self.mode]
        filtered_labels = {k: all_labels[k] for k in required_keys}

        if is_carcinoma:
            for k, v in filtered_labels.items():
                assert len(v) > 0, f"Missing required label {k} for slide"

        sup = sup_class(is_carcinoma, ignore=ignore, **filtered_labels)
        sup.ids, sup.ignore_ids = ids, ignore_ids
        return sup


def sorted_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """A slide's supervision rows in `get_nuclei` order (sorted by nucleus id)."""
    return df.sort_values("id")


def take_rows(df: pd.DataFrame, rows: NDArray[np.intp]) -> pd.DataFrame:
    """`df.iloc[rows]` for the (ascending) row positions of one slide.

    Contiguous rows (a slide's rows from its own file) are sliced, which does not copy
    the Arrow-backed columns; selecting rows of a multi-chunk Arrow column concatenates
    the whole column first.
    """
    if len(rows) > 0 and rows[-1] - rows[0] + 1 == len(rows):
        return df.iloc[rows[0] : rows[-1] + 1]
    return df.iloc[rows]


def slide_supervision(
    sources: list[pd.DataFrame],
    source_rows: list[dict[str, NDArray[np.intp]]],
    slide_id: str,
) -> pd.DataFrame:
    """The slide's rows of all the `sources`, inner-joined on the nucleus id and sorted by it.

    Raises `KeyError` if a source has no rows of the slide.
    """
    return sorted_by_id(
        reduce(
            lambda left, right: pd.merge(
                left, right, on=["slide_id", "id"], how="inner", validate="1:1"
            ),
            [
                take_rows(df, rows[slide_id])
                for df, rows in zip(sources, source_rows, strict=True)
            ],
        )
    )


def build_supervision(
    strategy: SupervisionStrategy,
    carcinoma_map: dict[str, bool],
    sup_dfs: dict[str, pd.DataFrame | None],
) -> DatasetSupervision:
    """Constructs Supervision for each slide based on the provided strategy and supervision data.

    `sup_dfs[IGNORE_KEY]` (optional) holds `ignore_label` for nuclei to exclude from
    supervision; it may cover only a subset of slides (incl. negative ones).
    """
    ignore_df = sup_dfs.get(IGNORE_KEY)
    sources = [df for k, df in sup_dfs.items() if k != IGNORE_KEY and df is not None]
    assert sources

    # Row positions of each slide (no copies of the dataframes); a slide's rows are
    # merged across the sources and sorted by id only when its supervision is built,
    # since a sorted copy of all the slides' rows does not fit in memory on large
    # datasets (e.g. iCAIRD: 131M training nuclei; the old code peaked at ~84 GB).
    source_rows = [df.groupby("slide_id", sort=False).indices for df in sources]
    ignore_rows = (
        ignore_df.groupby("slide_id", sort=False).indices
        if ignore_df is not None
        else {}
    )

    sup_map = {}
    empty_t = torch.empty(0, dtype=torch.float32)
    _, required_cols = strategy.STRATEGY_MAP[strategy.mode]

    for slide_id, is_carcinoma in tqdm(
        carcinoma_map.items(), desc="Building Supervision"
    ):
        labels = dict.fromkeys(required_cols, empty_t)

        ids = None
        if is_carcinoma:
            group = slide_supervision(sources, source_rows, slide_id)
            ids = IdsDigest.of(group["id"].tolist())
            for col in required_cols:
                if col in group.columns:
                    labels[col] = torch.from_numpy(group[col].to_numpy()).float()

        ignore, ignore_ids = None, None
        if slide_id in ignore_rows:
            assert ignore_df is not None
            group = sorted_by_id(take_rows(ignore_df, ignore_rows[slide_id]))
            ignore = torch.from_numpy(group["ignore_label"].to_numpy() == 1)
            ignore_ids = IdsDigest.of(group["id"].tolist())

        nuclei_sup = strategy.create(
            is_carcinoma, ignore=ignore, ids=ids, ignore_ids=ignore_ids, **labels
        )
        sup_map[slide_id] = SlideSupervision(int(is_carcinoma), nuclei_sup)

    return DatasetSupervision(sup_map)
