import numpy as np
import pytest
import torch
from scipy.spatial import KDTree
from torch.nn.attention.flex_attention import BlockMask

from nuclei_graph.data.block_mask import (
    batch_block_masks,
    create_block_mask_from_kdtree,
)


@pytest.fixture
def simple_points():
    points = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],  # Block 0
            [10.0, 10.0],
            [10.1, 10.1],  # Block 1
        ],
        dtype=np.float32,
    )
    tree = KDTree(points)
    return tree, points


@pytest.mark.parametrize(
    "block_size, k, n_unpadded, expected_counts",
    [
        (2, 2, 4, [1, 1]),
        (2, 1, 4, [1, 1]),
    ],
)
def test_create_block_mask_logic(
    simple_points, block_size, k, n_unpadded, expected_counts
):
    tree, points = simple_points
    mask = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=n_unpadded, k=k, block_size=block_size
    )

    kv_counts = mask.kv_num_blocks[0]
    kv_indices = mask.kv_indices[0]

    # each block should attend to itself only
    for q_block, expected in enumerate(expected_counts):
        assert kv_counts[q_block] == expected
        assert q_block in kv_indices[q_block].tolist()


def test_create_block_mask_shape(simple_points):
    tree, points = simple_points
    mask = create_block_mask_from_kdtree(tree, points, 4, k=2, block_size=2)

    assert isinstance(mask, BlockMask)
    assert mask.kv_num_blocks.shape == (1, 2)
    assert mask.kv_indices.shape[0] == 1
    assert mask.kv_indices.shape[1] == 2


def test_padding_validity():
    """Ensure padded points are ignored even if they are the closest neighbors."""
    points = np.array([[0.0], [0.1], [1.0], [1.05]], dtype=np.float32)
    n_unpadded = 3
    block_size = 1
    k = 2
    tree = KDTree(points)
    mask = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=n_unpadded, k=k, block_size=block_size
    )
    indices = mask.kv_indices[0, 2]
    max_valid_block = (n_unpadded // block_size) - 1

    is_valid = (indices <= max_valid_block) | (indices == -1)
    assert is_valid.all(), f"Found reference to padded block in indices: {indices}"
    assert 3 not in indices  # block index 3 corresponds to padded point


def test_batch_block_masks(simple_points):
    tree, points = simple_points

    mask1 = create_block_mask_from_kdtree(tree, points, 4, k=2, block_size=2)
    mask2 = create_block_mask_from_kdtree(tree, points, 4, k=2, block_size=2)
    batched = batch_block_masks([mask1, mask2])

    assert batched.kv_num_blocks.shape == (2, 1, 2)
    assert batched.kv_indices.shape[0] == 2

    assert torch.equal(batched.kv_num_blocks[0], mask1.kv_num_blocks)
    assert torch.equal(batched.kv_num_blocks[1], mask2.kv_num_blocks)


def test_batch_padding_logic():
    """Test batching masks with different numbers of KV neighbors."""
    pts_a = np.array([[0, 0], [10, 10]], dtype=np.float32)
    tree_a = KDTree(pts_a)
    mask_a = create_block_mask_from_kdtree(tree_a, pts_a, 2, k=1, block_size=1)

    pts_b = np.array([[0, 0], [0, 0]], dtype=np.float32)
    tree_b = KDTree(pts_b)
    mask_b = create_block_mask_from_kdtree(tree_b, pts_b, 2, k=2, block_size=1)

    batched = batch_block_masks([mask_a, mask_b])
    assert batched.kv_indices.shape[-1] == 2

    indices_a = batched.kv_indices[0, 0, 0]
    assert -1 in indices_a


@pytest.mark.parametrize("k", [1, 2])
def test_self_block_attention(simple_points, k):
    """Each block should attend to itself at minimum."""
    tree, points = simple_points
    mask = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=4, k=k, block_size=2
    )
    for q_block in range(2):
        assert q_block in mask.kv_indices[0, q_block].tolist()


def test_cross_block_attention():
    """Ensure blocks can attend to neighboring blocks if kNN picks points across blocks."""
    points = np.array(
        [
            [0.0, 0.0],
            [1.9, 0.0],
            [2.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    tree = KDTree(points)

    mask = create_block_mask_from_kdtree(tree, points, 4, k=2, block_size=2)

    assert 1 in mask.kv_indices[0, 0].tolist()
    assert 0 in mask.kv_indices[0, 1].tolist()
    assert mask.kv_num_blocks[0, 0] == len(set(mask.kv_indices[0, 0].tolist()) - {-1})
    assert mask.kv_num_blocks[0, 1] == len(set(mask.kv_indices[0, 1].tolist()) - {-1})


def test_block_deduplication():
    """Multiple point-to-point connections between blocks should result in a single block connection."""
    points = np.zeros((4, 2), dtype=np.float32)
    tree = KDTree(points)
    mask = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=4, k=4, block_size=2
    )

    assert mask.kv_num_blocks[0, 0] == 2

    indices = mask.kv_indices[0, 0, :2]
    assert torch.equal(indices.sort()[0], torch.tensor([0, 1], dtype=torch.int32))
