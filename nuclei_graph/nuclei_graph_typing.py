from typing import TypedDict

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


class Sample(TypedDict):
    x: Tensor
    pos: Tensor
    y: Tensor
    target_mask: Tensor
    block_mask: BlockMask


class Metadata(TypedDict):
    slide_id: str
    nuclei_ids: list[str]
    nuclei_count: int
    perm_inverse: Tensor


type PredictInput = tuple[Sample, list[Metadata]]

type PredictSample = tuple[Sample, Metadata]
