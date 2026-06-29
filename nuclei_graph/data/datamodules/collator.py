import math
from typing import Any, Final, cast

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from nuclei_graph.data.block_mask import (
    block_spatial_sort,
    create_ragged_block_quantized_knn_mask,
)
from nuclei_graph.nuclei_graph_typing import Batch, BatchMetadata, Sample, Targets


class GraphCollator:
    BUCKETS: Final[tuple[int, ...]] = (4096, 8192, 16384, 32768, 49152)

    def __init__(self, block_size: int = 128, k: int = 16, predict: bool = True):
        self.block_size = block_size
        self.k = k
        self.predict = predict

    def _pick_bucket(self, seq_len: int) -> int:
        for b in self.BUCKETS:
            if seq_len <= b:
                return b
        return math.ceil(seq_len / self.block_size) * self.block_size

    def _pad(self, x: Tensor, total_seq_len: int, value: float | int = 0) -> Tensor:
        pad_len = total_seq_len - x.shape[0]
        if pad_len <= 0:
            return x
        pad_shape = (pad_len, *x.shape[1:])
        pad = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
        return torch.cat((x, pad), dim=0)

    def _aggregate_metadata(
        self, batch: list[Sample], sort_indices_list: list[np.ndarray]
    ) -> BatchMetadata | None:
        first = batch[0].get("metadata")
        if not first:
            return None

        metadata_dict: dict[str, list[Any]] = {key: [] for key in first}

        for b, sort_indices in zip(batch, sort_indices_list, strict=True):
            meta = b.get("metadata")
            if meta is None:
                continue
            for key in first:
                value = meta.get(key)
                if key == "nuclei_ids" and value is not None:
                    value = value[sort_indices]
                metadata_dict[key].append(value)

        return cast("BatchMetadata", metadata_dict)

    def __call__(self, batch: list[Sample]) -> Batch:
        batch = [b for b in batch if len(b["pos"]) > 0]
        if not batch:
            raise ValueError("All samples in batch are empty.")

        has_bboxes = "bboxes" in batch[0]

        all_pos, all_features, all_knns = [], [], []
        all_labels_nuclei: list[Tensor] = []
        all_labels_graph: list[Tensor] = []
        all_sup_masks: list[Tensor] = []
        all_roi_masks: list[Tensor] = []
        all_bboxes: list[Tensor] = []
        all_sort_indices: list[np.ndarray] = []

        current_global_idx = 0
        for b in batch:
            n_nodes = len(b["pos"])
            pos_tensor = b["pos"]

            sort_indices = block_spatial_sort(
                pos_tensor.numpy(), self.block_size, global_offset=current_global_idx
            )
            all_sort_indices.append(sort_indices)
            sorted_pos = pos_tensor[sort_indices]

            actual_k = min(self.k, n_nodes)

            sorted_pos_np = sorted_pos.numpy()
            nbrs = NearestNeighbors(n_neighbors=actual_k, metric="euclidean")
            _, knn_np = nbrs.fit(sorted_pos_np).kneighbors(sorted_pos_np)

            knn = torch.from_numpy(knn_np)

            if actual_k < self.k:
                pad = torch.full(
                    (n_nodes, self.k - actual_k), -1, dtype=knn.dtype, device=knn.device
                )
                knn = torch.cat([knn, pad], dim=1)

            all_pos.append(sorted_pos)
            all_knns.append(knn)
            all_features.append(b["features"][sort_indices])
            all_sup_masks.append(b["sup_mask"][sort_indices])
            all_roi_masks.append(b["roi_mask"][sort_indices])

            if has_bboxes:
                all_bboxes.append(b["bboxes"][sort_indices])

            if not self.predict:
                assert b["labels"]["nuclei"] is not None
                all_labels_nuclei.append(b["labels"]["nuclei"][sort_indices])

                if b["labels"].get("graph") is not None:
                    assert b["labels"]["graph"] is not None
                    all_labels_graph.append(b["labels"]["graph"])

            current_global_idx += len(sorted_pos)

        target_seq_len = self._pick_bucket(current_global_idx)

        targets = Targets({"nuclei": None, "graph": None})
        if not self.predict:
            targets["nuclei"] = self._pad(torch.cat(all_labels_nuclei), target_seq_len)
            targets["graph"] = torch.cat(all_labels_graph) if all_labels_graph else None

        block_mask = create_ragged_block_quantized_knn_mask(
            all_knns, self.block_size, total_seq_len=target_seq_len
        )

        result: Batch = {
            "all_knns": all_knns,
            "block_size": self.block_size,
            "block_mask": block_mask,
            "pos": self._pad(torch.cat(all_pos), target_seq_len),
            "features": self._pad(torch.cat(all_features), target_seq_len),
            "sup_mask": self._pad(
                torch.cat(all_sup_masks), target_seq_len, value=False
            ),
            "roi_mask": self._pad(
                torch.cat(all_roi_masks), target_seq_len, value=False
            ),
            "seq_lens": torch.stack([b["seq_len"] for b in batch]).to(torch.int32),
            "labels": targets,
            "metadata": self._aggregate_metadata(batch, all_sort_indices),
        }
        if has_bboxes:
            result["bboxes"] = self._pad(torch.cat(all_bboxes), target_seq_len)
        return result
