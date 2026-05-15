import math

import torch
from torch import Tensor

from nuclei_graph.data.block_mask import create_dense_document_mask


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
    total_seq_len: int | None = None,
) -> dict:
    all_pos, all_features = [], []
    all_labels_nuclei, all_labels_graph, all_sup_masks = [], [], []

    current_global_idx = 0
    for b in batch:
        all_pos.append(torch.from_numpy(b["pos"]))
        all_features.append(torch.from_numpy(b["features"]))

        all_labels_nuclei.append(b["labels"]["nuclei"])
        if b["labels"]["graph"] is not None:
            all_labels_graph.append(b["labels"]["graph"])

        all_sup_masks.append(b["sup_mask"])
        current_global_idx += len(b["pos"])

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
        "block_mask": create_dense_document_mask(
            [len(pos) for pos in all_pos],
            block_size,
            device=all_pos[0].device,
            total_seq_len=target_seq_len,
        ),
        "pos": _pad_to_seq_len(torch.cat(all_pos), target_seq_len),
        "features": _pad_to_seq_len(torch.cat(all_features), target_seq_len),
        "labels": batched_labels,
        "sup_mask": _pad_to_seq_len(
            torch.cat(all_sup_masks), target_seq_len, value=False
        ),
        "seq_lens": torch.stack([b["seq_len"] for b in batch]).to(torch.int32),
    }


def predict_collate_fn(
    batch: list[dict], block_size: int, total_seq_len: int | None = None
) -> dict:

    all_pos, all_features, all_sup_masks = [], [], []

    current_global_idx = 0
    for b in batch:
        slide_dict = b["slide"]
        pos_array = slide_dict["pos"]

        all_pos.append(torch.from_numpy(pos_array))
        all_features.append(torch.from_numpy(slide_dict["features"]))
        all_sup_masks.append(slide_dict["sup_mask"])

        current_global_idx += len(pos_array)

    real_seq_len = current_global_idx
    target_seq_len = total_seq_len or pick_bucket(real_seq_len, block_size)

    if target_seq_len < real_seq_len:
        raise ValueError(
            f"total_seq_len ({target_seq_len}) must be >= packed length ({real_seq_len})"
        )

    return {
        "slide": {
            "block_mask": create_dense_document_mask(
                seq_lens_list=[len(pos) for pos in all_pos],
                block_size=block_size,
                device=all_pos[0].device,
                total_seq_len=target_seq_len,
            ),
            "pos": _pad_to_seq_len(torch.cat(all_pos), target_seq_len),
            "features": _pad_to_seq_len(torch.cat(all_features), target_seq_len),
            "sup_mask": _pad_to_seq_len(
                torch.cat(all_sup_masks), target_seq_len, value=False
            ),
            "seq_lens": torch.stack([b["slide"]["seq_len"] for b in batch]).to(
                torch.int32
            ),
        },
        "metadata": [b["metadata"] for b in batch],
    }
