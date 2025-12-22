import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


def dummy_mask_mod(
    q_block_idx: Tensor, kv_block_idx: Tensor, q_idx: Tensor, kv_idx: Tensor
) -> Tensor:
    return torch.ones_like(q_block_idx, dtype=torch.bool)  # attend all


def create_block_mask(
    kdtree: KDTree,
    points: NDArray[np.floating],
    n_points_unpadded: int,
    k: int,
    block_size: int,
) -> BlockMask:
    """Generates a single-item BlockMask from a KDTree and a corresponding point array.

    Padded points are excluded so that they neither attend to nor are attended by any key/value blocks.

    Args:
        kdtree: KDTree built over the points.
        points: A sorted (n, d) numpy array of positions, n must be divisible by block_size.
        k: Number of neighbors to query.
        n_points_unpadded: Number of points without the padding.
        block_size: Number of points per block.

    Returns:
        A BlockMask object with layouts:
            - kv_num_blocks: (1, num_blocks), number of key/value blocks per query block
            - kv_indices: (1, num_blocks, num_blocks), indices of key/value blocks
            - BLOCK_SIZE: (block_size, block_size)
        where num_blocks = n_points // block_size, n_points % block_size = 0
    """
    n_points = points.shape[0]
    assert n_points % block_size == 0
    num_blocks = n_points // block_size

    # 1. Build Block Adjacency Mask from a kNN Query
    # --------------------------------------------
    # query `k` neighbors for every valid (unpadded) point
    _, neighbor_indices = kdtree.query(points[:n_points_unpadded], k=k)

    # convert point indices to block indices
    q_block_ids = np.arange(n_points_unpadded) // block_size
    kv_block_ids = neighbor_indices // block_size

    # we have to filter out neighbors that point to the padded region
    valid_mask = neighbor_indices < n_points_unpadded

    adj_matrix = np.zeros((num_blocks, num_blocks), dtype=bool)
    # mark blocks as connected if any point in Q attends to any point in K
    adj_matrix[q_block_ids[:, None][valid_mask], kv_block_ids[valid_mask]] = True

    # 2. Convert to Sparse Format
    # --------------------------------------------
    # count how many KV blocks each Q block attends to
    kv_counts = adj_matrix.sum(axis=1)
    max_kv_len = kv_counts.max() if kv_counts.size > 0 else 0

    # initialize the output tensors (-1 for the padding)
    kv_num_blocks = torch.from_numpy(kv_counts).int().unsqueeze(0)  # (1, num_blocks)
    kv_indices = torch.full((1, num_blocks, max_kv_len), -1, dtype=torch.int32)

    # get coordinates of all connections (rows=Q, cols=KV)
    rows, cols = np.nonzero(adj_matrix)

    # to fill kv_indices[batch, row, slot] = col, we need a slot index for each connection
    cum_counts = np.cumsum(kv_counts)  # cumulative count of connections per block
    shifts = np.zeros_like(cum_counts)  # where each row starts in the flattened list
    shifts[1:] = cum_counts[:-1]  # shift right to get start indices
    global_idx = np.arange(len(rows))  # position in flattened list
    slot_idx = global_idx - shifts[rows]  # local_idx = global_idx - start_idx_of_row

    kv_indices[0, rows, slot_idx] = torch.from_numpy(cols).int()

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=dummy_mask_mod,
    )


def batch_block_masks(masks: list[BlockMask]) -> BlockMask:
    """Batch a list of single-item BlockMask objects into one batched BlockMask.

    Different neighbor counts (at the block level) are handled by padding.

    Note: All masks must have the same number of query blocks (sequence length).

    Args:
        masks: List of BlockMask objects.

    Returns:
        Batched BlockMask object with layouts:
            - kv_num_blocks: (b, 1, num_blocks)
            - kv_indices: (b, 1, num_blocks, max_num_blocks)
            - BLOCK_SIZE: (block_size, block_size)
        where:
            b = batch size,
            num_blocks = n // block_size,
            max_num_blocks = maximum number of KV blocks per query block across the batch.
    """
    first_mask = masks[0]
    block_size = first_mask.BLOCK_SIZE
    mask_mod = first_mask.mask_mod

    # concatenate along the batch dimension
    kv_num_blocks = torch.cat([m.kv_num_blocks for m in masks], dim=0)
    kv_indices_list = [m.kv_indices for m in masks]
    max_kv_len = max(t.shape[-1] for t in kv_indices_list)  # maximum neighbor count

    padded_kv_indices = []
    for kv_tensor in kv_indices_list:
        # pad to the maximum neighbor count in the batch
        pad_len = max_kv_len - kv_tensor.shape[-1]
        if pad_len > 0:
            padding = (0, pad_len)
            kv_tensor = torch.nn.functional.pad(kv_tensor, padding, "constant", -1)
        padded_kv_indices.append(kv_tensor)

    # result: (batch, num_blocks, max_neighbors_global)
    kv_indices = torch.cat(padded_kv_indices, dim=0)

    # unsqueeze for the heads dimension (B, H, Q, K)
    batched_mask = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks.unsqueeze(1),
        kv_indices=kv_indices.unsqueeze(1),
        full_kv_num_blocks=None,
        full_kv_indices=None,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
    )
    return batched_mask
