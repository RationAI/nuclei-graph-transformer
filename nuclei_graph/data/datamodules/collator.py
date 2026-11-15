import math
from typing import Any, Final, cast

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from nuclei_graph.data.block_mask import (
    BlockMask,
    block_spatial_sort,
    create_dense_document_mask,
    create_ragged_block_quantized_knn_mask,
    pack_and_shift_knn_indices,
)
from nuclei_graph.nuclei_graph_typing import (
    CONTEXT_MODES,
    Batch,
    BatchMetadata,
    Sample,
    Targets,
)


class GraphCollator:
    BUCKETS: Final[tuple[int, ...]] = (4096, 8192, 16384, 32768, 49152)

    def __init__(
        self,
        block_size: int = 128,
        k: int = 16,
        predict: bool = True,
        context: str = "knn",
        build_block_mask: bool = True,
    ):
        """Initializes the collator.

        Args:
            block_size: Block size of the attention block mask.
            k: Number of nearest neighbors of each nucleus.
            predict: Leave out the labels.
            context: Attention context: "knn", "dense" or "none".
            build_block_mask: Build the attention block mask here. Otherwise the batch
                leaves it `None` and `ensure_block_mask` builds it later, on the device
                of the batch: whole slides can have millions of nuclei, and a mask
                built in a dataloader worker is held in host memory by every worker
                and passed through shared memory.
        """
        assert context in CONTEXT_MODES, f"Invalid context: {context}"
        self.block_size = block_size
        self.k = k
        self.predict = predict
        self.context = context
        self.build_block_mask = build_block_mask

    def _pick_bucket(self, seq_len: int) -> int:
        for b in self.BUCKETS:
            if seq_len <= b:
                return b
        return math.ceil(seq_len / self.block_size) * self.block_size

    def _pad(self, x: Tensor, total_seq_len: int, value: float = 0) -> Tensor:
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

        has_features = batch[0].get("features") is not None
        has_bboxes = batch[0].get("bboxes") is not None

        all_pos, all_features, all_knns = [], [], []
        all_labels_nuclei: list[Tensor] = []
        all_labels_graph: list[Tensor] = []
        all_labels_pseudo: list[Tensor] = []
        all_sup_masks: list[Tensor] = []
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
            all_sup_masks.append(b["sup_mask"][sort_indices])

            if has_features:
                assert b["features"] is not None
                all_features.append(b["features"][sort_indices])
            if has_bboxes:
                assert b["bboxes"] is not None
                all_bboxes.append(b["bboxes"][sort_indices])

            if not self.predict:
                assert b["labels"]["nuclei"] is not None
                all_labels_nuclei.append(b["labels"]["nuclei"][sort_indices])

                if b["labels"].get("graph") is not None:
                    assert b["labels"]["graph"] is not None
                    all_labels_graph.append(b["labels"]["graph"])

                pseudo = b["labels"].get("pseudo")
                if pseudo is not None:
                    all_labels_pseudo.append(pseudo[sort_indices])

            current_global_idx += len(sorted_pos)

        target_seq_len = self._pick_bucket(current_global_idx)

        targets = Targets({"nuclei": None, "graph": None})
        if not self.predict:
            targets["nuclei"] = self._pad(torch.cat(all_labels_nuclei), target_seq_len)
            targets["graph"] = torch.cat(all_labels_graph) if all_labels_graph else None
            if all_labels_pseudo:
                assert len(all_labels_pseudo) == len(batch)
                targets["pseudo"] = self._pad(
                    torch.cat(all_labels_pseudo), target_seq_len, value=False
                )

        block_mask = (
            build_block_mask(all_knns, self.block_size, self.context, target_seq_len)
            if self.build_block_mask
            else None
        )

        global_neighbor_idx = pack_and_shift_knn_indices(all_knns, target_seq_len)

        return Batch(
            {
                "all_knns": all_knns,
                "global_neighbor_idx": global_neighbor_idx,  # (target_seq_len, k), -1 = invalid
                "block_size": self.block_size,
                "block_mask": block_mask,
                "context": self.context,
                "pos": self._pad(torch.cat(all_pos), target_seq_len),
                "features": self._pad(torch.cat(all_features), target_seq_len)
                if has_features
                else None,
                "bboxes": self._pad(torch.cat(all_bboxes), target_seq_len)
                if has_bboxes
                else None,
                "sup_mask": self._pad(
                    torch.cat(all_sup_masks), target_seq_len, value=False
                ),
                "seq_lens": torch.stack([b["seq_len"] for b in batch]).to(torch.int32),
                "labels": targets,
                "metadata": self._aggregate_metadata(batch, all_sort_indices),
            }
        )


def build_block_mask(
    all_knns: list[Tensor], block_size: int, context: str, total_seq_len: int
) -> BlockMask | None:
    """The attention block mask of the packed samples (`None` for context "none").

    Built on the device of `all_knns`, the per-sample k-NN indices (one row per
    nucleus).
    """
    if context == "knn":
        return create_ragged_block_quantized_knn_mask(
            all_knns, block_size, total_seq_len=total_seq_len
        )
    if context == "dense":
        return create_dense_document_mask(
            [len(knn) for knn in all_knns],
            block_size,
            device=all_knns[0].device,
            total_seq_len=total_seq_len,
        )
    return None


def ensure_block_mask(batch: Batch) -> Batch:
    """Builds the block mask of a batch whose collator left it out, in place.

    The mask is built on the device the batch is on (see `GraphCollator`), once:
    later calls with the same batch reuse it.
    """
    context = batch.get("context")
    if batch["block_mask"] is None and context in ("knn", "dense"):
        batch["block_mask"] = build_block_mask(
            batch["all_knns"],
            batch["block_size"],
            context,
            total_seq_len=len(batch["global_neighbor_idx"]),
        )
    return batch
