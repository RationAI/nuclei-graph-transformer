from typing import Any, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


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
    features: Tensor
    labels: Targets
    pos: Tensor
    sup_mask: Tensor
    roi_mask: Tensor
    seq_len: Tensor
    metadata: Metadata | None


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
    pos: Tensor
    features: Tensor
    sup_mask: Tensor
    roi_mask: Tensor
    seq_lens: Tensor
    labels: Targets
    metadata: BatchMetadata | None

    block_mask: NotRequired[Any]
