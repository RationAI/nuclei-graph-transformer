import math

import torch
from sklearn.neighbors import NearestNeighbors
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
) -> dict:
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")

    all_pos, all_features, all_knns, all_patches = [], [], [], []
    all_labels_nuclei, all_labels_graph, all_sup_masks = [], [], []

    current_global_idx = 0
    for b in batch:
        sort_indices = block_spatial_sort(
            b["pos"], block_size, global_offset=current_global_idx
        )
        sorted_pos = b["pos"][sort_indices]

        _, knn = nbrs.fit(sorted_pos).kneighbors(sorted_pos)

        all_pos.append(torch.from_numpy(sorted_pos))
        all_knns.append(torch.from_numpy(knn))
        all_features.append(torch.from_numpy(b["features"][sort_indices]))
        all_patches.append(b["patches"][sort_indices])

        all_labels_nuclei.append(b["labels"]["nuclei"][sort_indices])
        if b["labels"]["graph"] is not None:
            all_labels_graph.append(b["labels"]["graph"])

        all_sup_masks.append(b["sup_mask"][sort_indices])
        current_global_idx += len(sorted_pos)

    real_seq_len = current_global_idx
    target_seq_len = total_seq_len or pick_bucket(real_seq_len, block_size)
    if target_seq_len < real_seq_len:
        raise ValueError(
            f"total_seq_len ({target_seq_len}) must be >= packed length ({real_seq_len})"
        )

    batched_labels = {
        "nuclei": _pad_to_seq_len(torch.cat(all_labels_nuclei), target_seq_len),
        "graph": torch.cat(all_labels_graph) if all_labels_graph else None,
    }

    return {
        "block_mask": create_ragged_block_quantized_knn_mask(
            all_knns, block_size, total_seq_len=target_seq_len
        ),
        "pos": _pad_to_seq_len(torch.cat(all_pos), target_seq_len),
        "features": _pad_to_seq_len(torch.cat(all_features), target_seq_len),
        "patches": _pad_to_seq_len(torch.cat(all_patches), target_seq_len),
        "labels": batched_labels,
        "sup_mask": _pad_to_seq_len(
            torch.cat(all_sup_masks), target_seq_len, value=False
        ),
        "seq_lens": torch.stack([b["seq_len"] for b in batch]).to(torch.int32),
    }


def predict_collate_fn(
    batch: list[dict],
    block_size: int,
    k: int,
    total_seq_len: int | None = None,
) -> dict:
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")

    all_pos, all_features, all_patches, all_knns, all_sup_masks = [], [], [], [], []

    current_global_idx = 0
    for b in batch:
        slide_dict = b["slide"]

        sort_indices = block_spatial_sort(
            slide_dict["pos"], block_size, global_offset=current_global_idx
        )
        sorted_pos = slide_dict["pos"][sort_indices]
        _, knn = nbrs.fit(sorted_pos).kneighbors(sorted_pos)

        all_pos.append(torch.from_numpy(sorted_pos))
        all_knns.append(torch.from_numpy(knn))
        all_features.append(torch.from_numpy(slide_dict["features"][sort_indices]))
        all_patches.append(slide_dict["patches"][sort_indices])
        all_sup_masks.append(slide_dict["sup_mask"][sort_indices])

        b["metadata"]["nuclei_ids"] = b["metadata"]["nuclei_ids"][sort_indices]

        current_global_idx += len(sorted_pos)

    real_seq_len = current_global_idx
    target_seq_len = total_seq_len or pick_bucket(real_seq_len, block_size)
    if target_seq_len < real_seq_len:
        raise ValueError(
            f"total_seq_len ({target_seq_len}) must be >= packed length ({real_seq_len})"
        )

    return {
        "slide": {
            "block_mask": create_ragged_block_quantized_knn_mask(
                all_knns, block_size, total_seq_len=target_seq_len
            ),
            "pos": _pad_to_seq_len(torch.cat(all_pos), target_seq_len),
            "features": _pad_to_seq_len(torch.cat(all_features), target_seq_len),
            "patches": _pad_to_seq_len(torch.cat(all_patches), target_seq_len),
            "sup_mask": _pad_to_seq_len(
                torch.cat(all_sup_masks), target_seq_len, value=False
            ),
            "seq_lens": torch.stack([b["slide"]["seq_len"] for b in batch]).to(
                torch.int32
            ),
        },
        "metadata": [b["metadata"] for b in batch],
    }
