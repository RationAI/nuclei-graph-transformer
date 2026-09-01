from typing import NamedTuple, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


EMBEDDING_MODES = (
    "efd",
    "bbox",
    "spatial",
    "efd_spatial",
    "blank",
)
POOLING_MODES = ("max", "mean", "top_k")  # nuclei-to-tile pooling

MAX_CROP_PATCH_SIDE = 8192

# Target nuclei patch physical size — estimated patch is 48px * 0.2339um ~ 11.23um
TARGET_BBOX_CONTEXT_UM = 11.0


class Box(NamedTuple):
    lx: int
    ly: int
    rx: int
    ry: int

    @property
    def w(self) -> int:
        return self.rx - self.lx

    @property
    def h(self) -> int:
        return self.ry - self.ly


class SlideSize(NamedTuple):
    w: int
    h: int


class DecodedRegion(NamedTuple):
    array: NDArray[np.uint8]
    origin_x: int
    origin_y: int


class Metadata(TypedDict):
    slide_id: str

    # Tile-specific
    x: NotRequired[int]
    y: NotRequired[int]

    # Crop-specific
    slide_path: NotRequired[str]
    slide_nuclei_path: NotRequired[str]
    nuclei_ids: NotRequired[NDArray[np.str_]]


class Targets(TypedDict):
    nuclei: Tensor | None
    graph: Tensor | None


class Sample(TypedDict):
    features: Tensor | None
    labels: Targets
    pos: Tensor
    sup_mask: Tensor
    seq_len: Tensor
    metadata: Metadata | None
    bboxes: Tensor | None


class Outputs(TypedDict):
    graph: Tensor
    nuclei: Tensor
    attn_weights: Tensor


class BatchMetadata(TypedDict):
    slide_id: list[str]

    x: NotRequired[list[int]]
    y: NotRequired[list[int]]

    slide_path: NotRequired[list[str]]
    slide_nuclei_path: NotRequired[list[str]]
    nuclei_ids: NotRequired[list[NDArray[np.str_]]]


class Batch(TypedDict):
    all_knns: list[Tensor]
    block_size: int
    block_mask: BlockMask
    pos: Tensor
    features: Tensor | None
    sup_mask: Tensor
    seq_lens: Tensor
    labels: Targets
    metadata: BatchMetadata | None
    bboxes: Tensor | None
