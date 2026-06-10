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


class TileMetadata(TypedDict):
    slide: str
    x: int
    y: int


class TileGraph(TypedDict):
    features: Tensor
    labels: Targets
    pos: Tensor
    sup_mask: Tensor
    roi_mask: Tensor
    seq_len: Tensor
    metadata: TileMetadata


class CropMetadata(TypedDict):
    slide_id: str
    slide_path: str
    slide_nuclei_path: str
    nuclei_ids: NDArray[np.str_]


class Outputs(TypedDict):
    graph: Tensor
    nuclei: Tensor
    attn_weights: Tensor


class PredictCrop(TypedDict):
    slide: Crop
    metadata: CropMetadata


class Batch(TypedDict):
    block_mask: BlockMask
    features: Tensor
    pos: Tensor
    labels: Targets
    sup_mask: Tensor
    seq_lens: Tensor


class PredictBatch(TypedDict):
    slide: Batch
    metadata: list[CropMetadata]


class BatchMetadata(TypedDict):
    slide: list[str]
    x: list[int]
    y: list[int]


class GraphInputs(TypedDict):
    block_mask: BlockMask
    features: Tensor
    pos: Tensor
    sup_mask: Tensor
    seq_lens: Tensor


type LabeledSampleBatch = tuple[GraphInputs, Targets, BatchMetadata]
type UnlabeledSampleBatch = tuple[GraphInputs, BatchMetadata]
