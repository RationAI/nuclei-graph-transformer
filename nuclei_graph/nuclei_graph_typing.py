from typing import TypedDict

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
    indicator_mask: Tensor
    block_mask: BlockMask


type PredictInput = tuple[Sample, list[Metadata]]

type PredictSample = tuple[Sample, Metadata]

type Batch = list[Sample]

type PredictBatch = list[PredictSample]

type PartialConf = DictConfig

type Outputs = Tensor
