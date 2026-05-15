import math

import torch
import torch.nn.attention.flex_attention
from torch import Tensor
from torch.utils._pytree import tree_map_only


class _MaskMod:
    def __init__(self, doc_ids: Tensor) -> None:
        self.doc_ids = doc_ids

    def to(self, device: torch.device | str) -> "_MaskMod":
        return _MaskMod(self.doc_ids.to(device))

    def __call__(self, b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        # If the tokens don't belong to the same document, zero out the attention.
        return (self.doc_ids[q] >= 0) & (self.doc_ids[q] == self.doc_ids[kv])


class BlockMask(torch.nn.attention.flex_attention.BlockMask):
    def to(self, device: torch.device | str) -> "BlockMask":
        mapped_attributes = tree_map_only(
            (torch.Tensor, _MaskMod),
            lambda x: x.to(device),
            self.as_tuple(flatten=False),
        )
        return BlockMask(*mapped_attributes)


def create_dense_document_mask(
    seq_lens_list: list[int],
    block_size: int,
    device: torch.device,
    total_seq_len: int | None = None,
) -> BlockMask:
    """Creates a BlockMask for packed documents without spatial restrictions."""
    doc_ids_list = [
        torch.full((l,), i, dtype=torch.int32, device=device)
        for i, l in enumerate(seq_lens_list)
    ]
    doc_ids = torch.cat(doc_ids_list, dim=0)
    real_seq_len = doc_ids.shape[0]

    # Use target sequence length for block calculations
    if total_seq_len is None:
        total_seq_len = real_seq_len

    num_blocks = math.ceil(total_seq_len / block_size)

    # Pad doc_ids up to the block boundaries of the target_seq_len
    pad_len = num_blocks * block_size - real_seq_len
    if pad_len > 0:
        pad_tensor = torch.full((pad_len,), -1, dtype=torch.int32, device=device)
        padded_doc_ids = torch.cat([doc_ids, pad_tensor], dim=0)
    else:
        padded_doc_ids = doc_ids

    # === 2. Build Global Block Adjacency Matrix (NumBlocks x NumBlocks) ===
    block_starts = torch.arange(num_blocks, device=device) * block_size
    block_ends = block_starts + block_size - 1

    start_docs = padded_doc_ids[block_starts]
    end_docs = padded_doc_ids[block_ends]

    # Two blocks can attend to each other if their document ranges overlap
    start_i = start_docs.unsqueeze(1)
    end_i = end_docs.unsqueeze(1)
    start_j = start_docs.unsqueeze(0)
    end_j = end_docs.unsqueeze(0)

    # Overlap logic: max(start_i, start_j) <= min(end_i, end_j)
    adj_matrix = torch.max(start_i, start_j) <= torch.min(end_i, end_j)

    # Ignore padding blocks
    valid_blocks = (start_docs >= 0).unsqueeze(1) & (start_docs >= 0).unsqueeze(0)
    adj_matrix = adj_matrix & valid_blocks

    kv_num_blocks = adj_matrix.sum(dim=-1, dtype=torch.int32)

    # === 3. Compress to Dense KV Indices ===
    col_indices = (
        torch.arange(num_blocks, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(num_blocks, num_blocks)
    )

    masked_col_indices = torch.where(adj_matrix, col_indices, num_blocks + 1)
    sorted_indices, _ = masked_col_indices.sort(dim=-1)

    kv_indices = torch.where(
        sorted_indices > num_blocks,
        torch.tensor(-1, dtype=torch.int32, device=device),
        sorted_indices.to(torch.int32),
    )

    # === 4. Optimize Fast Path (Pure vs Mixed Blocks) ===
    is_pure_block = start_docs == end_docs

    valid_kv_mask = kv_indices >= 0
    safe_kv_indices = torch.where(valid_kv_mask, kv_indices, 0)
    is_kv_pure = is_pure_block[safe_kv_indices]

    mixed_q_mask = ~is_pure_block
    mixed_kv_mask = valid_kv_mask & (~is_kv_pure)

    full_kv_indices = kv_indices.clone()
    full_kv_indices.masked_fill_(mixed_q_mask.unsqueeze(-1), -1)
    full_kv_indices.masked_fill_(mixed_kv_mask, -1)

    sort_keys = torch.where(full_kv_indices == -1, num_blocks + 1, full_kv_indices)
    sorted_full_indices, _ = sort_keys.sort(dim=-1)

    full_kv_indices = torch.where(
        sorted_full_indices > num_blocks,
        torch.tensor(-1, dtype=torch.int32, device=device),
        sorted_full_indices.to(torch.int32),
    )

    full_kv_num_blocks = (full_kv_indices != -1).sum(dim=-1, dtype=torch.int32)

    # === 5. Return the highly optimized BlockMask ===
    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks.view(1, 1, num_blocks),
        kv_indices=kv_indices.view(1, 1, num_blocks, num_blocks),
        full_kv_num_blocks=full_kv_num_blocks.view(1, 1, num_blocks),
        full_kv_indices=full_kv_indices.view(1, 1, num_blocks, num_blocks),
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=_MaskMod(padded_doc_ids[:total_seq_len]),
        seq_lengths=(total_seq_len, total_seq_len),
    )
