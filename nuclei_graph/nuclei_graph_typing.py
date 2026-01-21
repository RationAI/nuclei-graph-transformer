from typing import TypedDict

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


class Sample(TypedDict):
    x: Tensor
    pos: Tensor
    y: Tensor
    sup_mask: Tensor
    block_mask: BlockMask
    num_points: int | Tensor


class Metadata(TypedDict):
    slide_id: str
    nuclei_ids: list[str]
    perm_inverse: Tensor


type PredictInput = tuple[Sample, list[Metadata]]

type PredictSample = tuple[Sample, Metadata]
