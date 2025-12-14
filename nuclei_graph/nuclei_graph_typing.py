from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


class Metadata(TypedDict):
    slide_id: str
    slide_path: str
    slide_nuclei_path: str
    nuclei_count: int
    perm_inverse: Tensor


class Sample(TypedDict):
    x: Tensor
    pos: Tensor
    y: Tensor
    annot_mask: Tensor
    block_mask: BlockMask


type Neighbor = tuple[int, float]  # (node_idx, edge_weight)

type AdjacencyGraph = list[list[Neighbor]]

type PointArray = NDArray[np.float32]

type PredictInput = tuple[Sample, list[Metadata]]

type PredictSample = tuple[Sample, Metadata]

type Batch = list[Sample]

type PredictBatch = list[PredictSample]

type PartialConf = DictConfig

type Outputs = Tensor
