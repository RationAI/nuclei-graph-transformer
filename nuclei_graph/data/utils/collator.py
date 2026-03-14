from collections.abc import Iterable

import torch
from nuclei_graph.data.block_mask import batch_block_masks

from nuclei_graph.nuclei_graph_typing import (
    Batch,
    Crop,
    PredictBatch,
    PredictSlide,
    Targets,
)


def collate_fn(batch: Iterable[Crop]) -> Batch:
    batch = list(batch)

    graph_targets = [b["y"]["graph"] for b in batch if b["y"]["graph"] is not None]
    batched_y: Targets = {
        "graph": torch.stack(graph_targets, dim=0) if graph_targets else None,
        "nuclei": torch.cat([b["y"]["nuclei"] for b in batch], dim=0),  # variable size
    }
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "pos": torch.stack([b["pos"] for b in batch], dim=0),
        "y": batched_y,
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
