from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


class Targets(TypedDict):
    nuclei: Tensor | None
    graph: Tensor | None


class Crop(TypedDict):
    features: NDArray[np.float32]
    labels: Targets
    pos: NDArray[np.float32]
    sup_mask: Tensor
    seq_len: Tensor


class Metadata(TypedDict):
    slide_id: str
    slide_path: str
    slide_nuclei_path: str
    nuclei_ids: NDArray[np.str_]


class Outputs(TypedDict):
    graph: Tensor
    nuclei: Tensor
    attn_weights: Tensor


Slide = Crop  # a full-slide Crop


class PredictSlide(TypedDict):
    slide: Slide
    metadata: Metadata


class Batch(TypedDict):
    block_mask: BlockMask
    features: Tensor
    pos: Tensor
    labels: Targets
    sup_mask: Tensor
    seq_lens: Tensor


class PredictBatch(TypedDict):
    slide: Batch
    metadata: list[Metadata]
