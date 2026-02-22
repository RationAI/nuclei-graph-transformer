from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


if TYPE_CHECKING:
    from nuclei_graph.data.supervision import NucleiSupervision


@dataclass(frozen=True)
class SlideSupervision:
    slide_label: int
    nuclei_supervision: NucleiSupervision


@dataclass(frozen=True)
class DatasetSupervision:
    supervision_map: dict[str, SlideSupervision]


class Crop(TypedDict):
    x: Tensor  # node features
    pos: Tensor  # positional features (coordinates + rotation)
    y: Tensor  # supervised labels
    sup_mask: Tensor  # supervision mask
    block_mask: BlockMask  # attention mask


class Metadata(TypedDict):
    slide_id: str
    slide_nuclei_path: str
    keep_indices: Tensor
    perm_inverse: Tensor


Slide = Crop  # a full-slide crop


class PredictSlide(TypedDict):
    slide: Slide
    metadata: Metadata


Batch = Slide  # batched slides


class PredictBatch(TypedDict):
    slides: Batch
    metadata: list[Metadata]
