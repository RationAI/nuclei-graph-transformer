import torch
from sklearn.neighbors import NearestNeighbors

from nuclei_graph.data.block_mask import (
    block_spatial_sort,
    create_ragged_block_quantized_knn_mask,
)


def supervised_collate_fn(batch: list[dict], block_size: int, k: int) -> dict:
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")

    all_pos, all_features, all_knns = [], [], []
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

        all_labels_nuclei.append(b["labels"]["nuclei"][sort_indices])
        if b["labels"]["graph"] is not None:
            all_labels_graph.append(b["labels"]["graph"])

        all_sup_masks.append(b["sup_mask"][sort_indices])
        current_global_idx += len(sorted_pos)

    batched_labels = {
        "nuclei": torch.cat(all_labels_nuclei),
        "graph": torch.cat(all_labels_graph) if all_labels_graph else None,
    }

    return {
        "block_mask": create_ragged_block_quantized_knn_mask(all_knns, block_size),
        "pos": torch.cat(all_pos),
        "features": torch.cat(all_features),
        "labels": batched_labels,
        "sup_mask": torch.cat(all_sup_masks),
        "seq_lens": torch.stack([b["seq_len"] for b in batch]).to(torch.int32),
    }


def predict_collate_fn(batch: list[dict], block_size: int, k: int) -> dict:
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")

    all_pos, all_features, all_knns, all_sup_masks = [], [], [], []

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
        all_sup_masks.append(slide_dict["sup_mask"][sort_indices])

        b["metadata"]["nuclei_ids"] = b["metadata"]["nuclei_ids"][sort_indices]

        current_global_idx += len(sorted_pos)

    return {
        "slide": {
            "block_mask": create_ragged_block_quantized_knn_mask(all_knns, block_size),
            "pos": torch.cat(all_pos),
            "features": torch.cat(all_features),
            "sup_mask": torch.cat(all_sup_masks),
            "seq_lens": torch.stack([b["slide"]["seq_len"] for b in batch]).to(
                torch.int32
            ),
        },
        "metadata": [b["metadata"] for b in batch],
    }
