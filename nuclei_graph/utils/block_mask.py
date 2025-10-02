import numpy as np
import torch
from scipy.spatial import KDTree
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


def dummy_mask_mod(
    q_block_idx: Tensor, kv_block_idx: Tensor, q_idx: Tensor, kv_idx: Tensor
) -> Tensor:
    return torch.ones_like(q_block_idx, dtype=torch.bool)


def create_single_block_mask_from_kdtree(
    kdtree: KDTree,
    points: np.ndarray,
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

    # build block adjacency mask
    mask = np.zeros((num_blocks, num_blocks), dtype=bool)
    for q_idx in range(n_points_unpadded):
        _, neighbor_indices = kdtree.query(points[q_idx], k=k)
        q_block = q_idx // block_size
        for n_idx in neighbor_indices:
            kv_block = n_idx // block_size
            mask[q_block, kv_block] = True

    # convert boolean mask to per-row global indices
    indices_list = [np.where(row)[0].tolist() for row in mask]
    num_blocks_per_row = [len(lst) for lst in indices_list]

    kv_num_blocks = torch.tensor([num_blocks_per_row], dtype=torch.int32)
    kv_indices = torch.full((1, num_blocks, num_blocks), -1, dtype=torch.int32)

    for i, lst in enumerate(indices_list):
        if lst:
            kv_indices[0, i, : len(lst)] = torch.tensor(lst, dtype=torch.int32)

    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=dummy_mask_mod,
    )
    return block_mask


def batch_block_masks(masks: list[BlockMask]) -> BlockMask:
    """Batch a list of single-item BlockMask objects into one batched BlockMask.

    Note: All masks must have the same number of query blocks (`block_size`).

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

    # concatenate kv_num_blocks along the batch dimension
    kv_num_blocks = torch.cat([m.kv_num_blocks for m in masks], dim=0)

    # pad the kv_indices and concatenate
    kv_indices_list = [m.kv_indices for m in masks]
    max_kv_len = max(t.shape[-1] for t in kv_indices_list)

    padded_kv_indices = []
    for kv_tensor in kv_indices_list:
        pad_len = max_kv_len - kv_tensor.shape[-1]
        if pad_len > 0:
            padding = (0, pad_len)
            kv_tensor = torch.nn.functional.pad(kv_tensor, padding, "constant", -1)
        padded_kv_indices.append(kv_tensor)
    kv_indices = torch.cat(padded_kv_indices, dim=0)

    batched_mask = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks.unsqueeze(1),
        kv_indices=kv_indices.unsqueeze(1),
        full_kv_num_blocks=None,
        full_kv_indices=None,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
    )
    return batched_mask
