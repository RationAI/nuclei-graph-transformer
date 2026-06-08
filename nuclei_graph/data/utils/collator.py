import math
import time
from typing import Any

import torch
from torch import Tensor

from nuclei_graph.data.block_mask import (
    block_spatial_sort,
    create_ragged_block_quantized_knn_mask,
)


BUCKETS = [4096, 8192, 16384, 32768, 49152]


def pick_bucket(seq_len, block_size=128):
    for b in BUCKETS:
        if seq_len <= b:
            return b
    return math.ceil(seq_len / block_size) * block_size


def _pad_to_seq_len(x: Tensor, total_seq_len: int, value: float | int = 0) -> Tensor:
    pad_len = total_seq_len - x.shape[0]
    if pad_len == 0:
        return x
    pad_shape = (pad_len, *x.shape[1:])
    pad = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat((x, pad), dim=0)


def supervised_collate_fn(
    batch: list[dict],
    block_size: int,
    k: int,
    total_seq_len: int | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor | None], dict[str, list[Any]]]:
    batch = [b for b in batch if len(b["pos"]) > 0]

    if not batch:
        raise ValueError("All samples in batch are empty.")

    all_pos, all_features, all_knns = [], [], []
    all_labels_nuclei, all_labels_graph, all_sup_masks, all_roi_masks = [], [], [], []

    current_global_idx = 0
    for b in batch:
        n_nodes = len(b["pos"])

        sort_indices = block_spatial_sort(
            b["pos"].numpy(),
            block_size,
            global_offset=current_global_idx,
        )
        sorted_pos = b["pos"][sort_indices]

        actual_k = min(k, n_nodes)

        dist_matrix = torch.cdist(sorted_pos, sorted_pos)
        knn = dist_matrix.topk(actual_k, largest=False).indices

        if actual_k < k:
            pad = torch.full(
                (n_nodes, k - actual_k), -1, dtype=knn.dtype, device=knn.device
            )
            knn = torch.cat([knn, pad], dim=1)

        all_pos.append(sorted_pos)
        all_knns.append(knn)
        all_features.append(b["features"][sort_indices])

        all_labels_nuclei.append(b["labels"]["nuclei"][sort_indices])
        if b["labels"]["graph"] is not None:
            all_labels_graph.append(b["labels"]["graph"])

        all_sup_masks.append(b["sup_mask"][sort_indices])
        all_roi_masks.append(b["roi_mask"][sort_indices])
        current_global_idx += len(sorted_pos)

    real_seq_len = current_global_idx
    target_seq_len = total_seq_len or pick_bucket(real_seq_len, block_size)

    batch_metadata = {
        "slide": [b["metadata"]["slide"] for b in batch],
        "x": [b["metadata"]["x"] for b in batch],
        "y": [b["metadata"]["y"] for b in batch],
    }

    inputs = {
        "all_knns": all_knns,
        "block_size": block_size,
        "pos": _pad_to_seq_len(torch.cat(all_pos), target_seq_len),
        "features": _pad_to_seq_len(torch.cat(all_features), target_seq_len),
        "sup_mask": _pad_to_seq_len(
            torch.cat(all_sup_masks), target_seq_len, value=False
        ),
        "roi_mask": _pad_to_seq_len(
            torch.cat(all_roi_masks), target_seq_len, value=False
        ),
        "seq_lens": torch.stack([b["seq_len"] for b in batch]).to(torch.int32),
    }

    targets = {
        "nuclei": _pad_to_seq_len(torch.cat(all_labels_nuclei), target_seq_len)
        if all_labels_nuclei
        else None,
        "graph": torch.cat(all_labels_graph) if all_labels_graph else None,
    }

    return inputs, targets, batch_metadata


def predict_collate_fn(
    batch: list[dict],
    block_size: int,
    k: int,
    total_seq_len: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    all_pos, all_features, all_knns, all_sup_masks, all_roi_masks = [], [], [], [], []

    current_global_idx = 0
    for b in batch:
        n_nodes = len(b["pos"])
        sort_indices = block_spatial_sort(
            b["pos"].numpy(), block_size, global_offset=current_global_idx
        )
        sorted_pos = b["pos"][sort_indices]
        actual_k = min(k, n_nodes)

        dist_matrix = torch.cdist(sorted_pos, sorted_pos)
        knn = dist_matrix.topk(actual_k, largest=False).indices

        if actual_k < k:
            pad = torch.full(
                (n_nodes, k - actual_k), -1, dtype=knn.dtype, device=knn.device
            )
            knn = torch.cat([knn, pad], dim=1)

        all_pos.append(sorted_pos)
        all_knns.append(knn)
        all_features.append(b["features"][sort_indices])
        all_sup_masks.append(b["sup_mask"][sort_indices])
        all_roi_masks.append(b["roi_mask"][sort_indices])

        current_global_idx += len(sorted_pos)

    real_seq_len = current_global_idx
    target_seq_len = total_seq_len or pick_bucket(real_seq_len, block_size)
    batch_metadata = {
        "slide": [b["metadata"]["slide"] for b in batch],
        "x": [b["metadata"]["x"] for b in batch],
        "y": [b["metadata"]["y"] for b in batch],
    }

    inputs = {
        "all_knns": all_knns,
        "block_size": block_size,
        "pos": _pad_to_seq_len(torch.cat(all_pos), target_seq_len),
        "features": _pad_to_seq_len(torch.cat(all_features), target_seq_len),
        "sup_mask": _pad_to_seq_len(
            torch.cat(all_sup_masks), target_seq_len, value=False
        ),
        "roi_mask": _pad_to_seq_len(
            torch.cat(all_roi_masks), target_seq_len, value=False
        ),
        "seq_lens": torch.stack([b["seq_len"] for b in batch]).to(torch.int32),
    }

    return inputs, batch_metadata
