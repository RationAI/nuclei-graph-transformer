from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from ray.data import Dataset as RayDatasetBase
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch_geometric.transforms import BaseTransform


class RayDataset[T](RayDatasetBase):
    """Type alias for Ray Dataset containing elements of type T."""


class NucleusRecord(TypedDict):
    slide_id: str
    tile_x: int
    tile_y: int
    polygon: list[list[float]]
    centroid: list[float]
    nucleus_id: str


class TilePolygonRecord(TypedDict):
    slide_id: str
    tile_x: int
    tile_y: int
    polygons: list[list[list[float]]]


class TileRecord(TypedDict):
    tile_x: int
    tile_y: int
    path: str
    slide_id: str
    level: int
    tile_extent_x: int
    tile_extent_y: int
    scale_factor: float


class SlideRecord(TypedDict):
    path: str
    extent_x: int
    extent_y: int
    tile_extent_x: int
    tile_extent_y: int
    stride_x: int
    stride_y: int
    mpp_x: float
    mpp_y: float
    level: int
    downsample: float
    slide_id: str
    scale_factor: float


class Metadata(TypedDict):
    slide_id: str
    slide_mrxs_path: str
    slide_nuclei_path: str
    nuclei_count: int
    perm_inverse: Tensor


class Sample(TypedDict):
    x: Tensor
    pos: Tensor
    y: Tensor
    annot_mask: Tensor
    block_mask: BlockMask


Transforms = list[BaseTransform] | BaseTransform

type TileCoords = tuple[int, int]

type Neighbor = tuple[int, float]  # (node_idx, edge_weight)

type AdjacencyGraph = list[list[Neighbor]]

type PointArray = NDArray[np.float32]

type PredictInput = tuple[Sample, list[Metadata]]

type PredictSample = tuple[Sample, Metadata]

type Batch = list[Sample]

type PredictBatch = list[PredictSample]

type PartialConf = DictConfig

type Outputs = Tensor

Region = list[tuple[int, int]]
