from typing import TypedDict

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


class Crop(TypedDict):
    x: Tensor  # node features
    pos: Tensor  # positional features
    y: Tensor  # supervised labels
    sup_mask: Tensor  # supervision mask
    block_mask: BlockMask  # attention mask
    seq_len: Tensor  # unpadded sequence length


class Metadata(TypedDict):
    slide_id: str
    slide_nuclei_path: str
    slide_path: str
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
