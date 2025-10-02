from typing import TypeAlias, TypedDict

from omegaconf import DictConfig
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch_geometric.transforms import BaseTransform


Transforms = list[BaseTransform] | BaseTransform


class Metadata(TypedDict):
    slide_id: str
    slide_tiff_path: str
    raw_cells_path: str
    nuclei_count: int
    perm_inverse: Tensor


class Sample(TypedDict):
    x: Tensor
    pos: Tensor
    y: Tensor
    annot_mask: Tensor
    block_mask: BlockMask


PredictInput: TypeAlias = tuple[Sample, list[Metadata]]

PredictSample: TypeAlias = tuple[Sample, Metadata]

Batch: TypeAlias = list[Sample]

PredictBatch: TypeAlias = list[PredictSample]

PartialConf: TypeAlias = DictConfig

Outputs: TypeAlias = Tensor

Region = list[tuple[int, int]]
