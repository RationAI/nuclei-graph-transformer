from collections.abc import Iterable

import torch
from nuclei_graph.data.block_mask import batch_block_masks

from nuclei_graph.nuclei_graph_typing import Batch, PredictBatch, PredictSample, Sample


def collate_fn(batch: Iterable[Sample]) -> Batch:
    batch = list(batch)
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "pos": torch.stack([b["pos"] for b in batch], dim=0),
        "y": torch.cat([b["y"] for b in batch], dim=0),  # variable-length tensors
        "wsl_masks": {
            "sup_mask": torch.stack([b["wsl_masks"]["sup_mask"] for b in batch], dim=0),
            "ignore_mask": torch.stack(
                [b["wsl_masks"]["ignore_mask"] for b in batch], dim=0
            ),
        },
        "num_points": torch.tensor([b["num_points"] for b in batch], dtype=torch.long),
        "block_mask": batch_block_masks([b["block_mask"] for b in batch]),
    }


def collate_fn_predict(batch: Iterable[PredictSample]) -> PredictBatch:
    batch = list(batch)
    return {
        "items": collate_fn([b["item"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }
