from collections.abc import Iterable

import torch
from nuclei_graph.data.block_mask import batch_block_masks

from nuclei_graph.nuclei_graph_typing import Batch, Crop, PredictBatch, PredictSlide


def collate_fn(batch: Iterable[Crop]) -> Batch:
    batch = list(batch)
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "pos": torch.stack([b["pos"] for b in batch], dim=0),
        "y": torch.cat([b["y"] for b in batch], dim=0),  # variable-length tensors
        "sup_mask": torch.stack([b["sup_mask"] for b in batch], dim=0),
        "block_mask": batch_block_masks([b["block_mask"] for b in batch]),
        "seq_len": torch.tensor([b["seq_len"] for b in batch], dtype=torch.int32),
    }


def collate_fn_predict(batch: Iterable[PredictSlide]) -> PredictBatch:
    batch = list(batch)
    return {
        "slides": collate_fn([b["slide"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }
