from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


@dataclass(frozen=True)
class SlideSupervision:
    slide_label: int
    annot_labels: Tensor | None = None  # sorted by nucleus id; None for negative slides
    cam_labels: Tensor | None = None  # sorted by nucleus id; None for negative slides


@dataclass(frozen=True)
class DatasetSupervision:
    sup_map: dict[str, SlideSupervision]


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
    batch: Batch
    metadata: list[Metadata]
