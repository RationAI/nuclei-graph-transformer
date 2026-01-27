from typing import TypedDict

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


class WSLMasks(TypedDict):
    sup_mask: Tensor
    ignore_mask: Tensor


class Sample(TypedDict):
    x: Tensor
    pos: Tensor
    y: Tensor
    wsl_masks: WSLMasks
    num_points: int
    block_mask: BlockMask


class Batch(TypedDict):
    x: Tensor
    pos: Tensor
    y: Tensor
    wsl_masks: WSLMasks
    num_points: Tensor
    block_mask: BlockMask


class Metadata(TypedDict):
    slide_id: str
    slide_nuclei_path: str
    nuclei_ids: list[str]
    perm_inverse: Tensor


class PredictSample(TypedDict):
    item: Sample
    metadata: Metadata


class PredictBatch(TypedDict):
    items: Batch
    metadata: list[Metadata]
