from typing import TypedDict

from numpy.typing import NDArray
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


class Targets(TypedDict):
    graph: Tensor | None
    nuclei: Tensor | None


class Crop(TypedDict):
    x: Tensor  # node features
    pos: Tensor  # positional features (coordinates + rotation)
    y: Targets  # labels
    sup_mask: Tensor  # supervision mask
    block_mask: BlockMask  # attention mask
    seq_len: Tensor  # unpadded sequence length


class Metadata(TypedDict):
    slide_id: str
    slide_path: str
    slide_nuclei_path: str
    perm_inverse: Tensor
    nuclei_ids: NDArray


class Outputs(TypedDict):
    graph: Tensor
    nuclei: Tensor
    attn_weights: Tensor


Slide = Crop  # a full-slide Crop


class PredictSlide(TypedDict):
    slide: Slide
    metadata: Metadata


Batch = Slide  # batched slides


class PredictBatch(TypedDict):
    slides: Batch
    metadata: list[Metadata]
