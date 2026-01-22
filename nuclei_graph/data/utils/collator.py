import torch

from nuclei_graph.data.block_mask import batch_block_masks
from nuclei_graph.nuclei_graph_typing import PredictInput, PredictSample, Sample


def collate_fn(batch: list[Sample]) -> Sample:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "pos": torch.stack([b["pos"] for b in batch], dim=0),
        "y": torch.cat([b["y"] for b in batch], dim=0),  # variable-length tensors
        "sup_mask": torch.stack([b["sup_mask"] for b in batch], dim=0),
        "ignore_mask": torch.stack([b["ignore_mask"] for b in batch], dim=0),
        "num_points": torch.tensor([b["num_points"] for b in batch], dtype=torch.long),
        "block_mask": batch_block_masks([b["block_mask"] for b in batch]),
    }


def collate_fn_predict(batch: list[PredictSample]) -> PredictInput:
    items, metadata = zip(*batch, strict=True)
    return collate_fn(list(items)), list(metadata)
