from typing import TypedDict

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


class Sample(TypedDict):
    x: Tensor
    pos: Tensor
    y: Tensor
    sup_mask: Tensor
    ignore_mask: Tensor
    num_points: int | Tensor
    block_mask: BlockMask


class Metadata(TypedDict):
    slide_id: str
    slide_nuclei_path: str
    nuclei_ids: list[str]
    perm_inverse: Tensor


class PredictSample(TypedDict):
    item: Sample
    metadata: Metadata


class PredictInput(TypedDict):
    item: Sample
    metadata: list[Metadata]
