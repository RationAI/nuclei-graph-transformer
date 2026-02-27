import numpy as np
import pytest
import torch
from scipy.spatial import KDTree
from torch.nn.attention.flex_attention import BlockMask

from nuclei_graph.data.block_mask import (
    batch_block_masks,
    create_block_mask_from_kdtree,
)


PointsFixture = tuple[KDTree, np.ndarray]


@pytest.fixture
def simple_points() -> PointsFixture:
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
    simple_points: PointsFixture,
    block_size: int,
    k: int,
    n_unpadded: int,
    expected_counts: list[int],
) -> None:
    tree, points = simple_points
    mask = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=n_unpadded, k=k, block_size=block_size
    )

    kv_counts = mask.kv_num_blocks[0, 0]
    kv_indices = mask.kv_indices[0, 0]

    # each block should attend to itself only
    for q_block, expected in enumerate(expected_counts):
        assert kv_counts[q_block] == expected
        assert q_block in kv_indices[q_block].tolist()


def test_create_block_mask_shape(simple_points: PointsFixture) -> None:
    tree, points = simple_points
    mask = create_block_mask_from_kdtree(tree, points, 4, k=2, block_size=2)

    assert isinstance(mask, BlockMask)
    assert mask.kv_num_blocks.shape == (1, 1, 2)
    assert mask.kv_indices.shape[0] == 1
    assert mask.kv_indices.shape[1] == 1


def test_padding_validity() -> None:
    """Ensure padded points are ignored even if they are the closest neighbors."""
    points = np.array([[0.0], [0.1], [1.0], [1.05]], dtype=np.float32)
    n_unpadded = 3
    block_size = 1
    k = 2
    tree = KDTree(points)
    mask = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=n_unpadded, k=k, block_size=block_size
    )
    # check that no query block references block 3 (the padded block)
    num_valid_blocks = (n_unpadded + block_size - 1) // block_size

    for q_block in range(num_valid_blocks):
        indices = mask.kv_indices[0, 0, q_block]
        valid_indices = indices[indices != -1]

        # all valid indices must be < num_valid_blocks (i.e., blocks 0, 1, 2)
        assert torch.all(valid_indices < num_valid_blocks), (
            f"Query block {q_block} references padded block: {indices.tolist()}"
        )
        assert 3 not in valid_indices.tolist(), (
            f"Query block {q_block} should not attend to padded block 3"
        )


def test_batch_block_masks(simple_points: PointsFixture) -> None:
    tree, points = simple_points

    mask1 = create_block_mask_from_kdtree(tree, points, 4, k=2, block_size=2)
    mask2 = create_block_mask_from_kdtree(tree, points, 4, k=2, block_size=2)
    batched = batch_block_masks([mask1, mask2])

    assert batched.kv_num_blocks.shape == (2, 1, 2)
    assert batched.kv_indices.shape[0] == 2

    assert torch.equal(batched.kv_num_blocks[0, 0], mask1.kv_num_blocks[0, 0])
    assert torch.equal(batched.kv_num_blocks[1, 0], mask2.kv_num_blocks[0, 0])


def test_batch_padding_logic() -> None:
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
def test_self_block_attention(simple_points: PointsFixture, k: int) -> None:
    """Each block should attend to itself at minimum."""
    tree, points = simple_points
    mask = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=4, k=k, block_size=2
    )
    for q_block in range(2):
        assert q_block in mask.kv_indices[0, 0, q_block].tolist()


def test_cross_block_attention() -> None:
    """Blocks can attend to neighboring blocks if kNN picks points across blocks."""
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

    assert 1 in mask.kv_indices[0, 0, 0].tolist()
    assert 0 in mask.kv_indices[0, 0, 1].tolist()
    assert mask.kv_num_blocks[0, 0, 0] == len(
        set(mask.kv_indices[0, 0, 0].tolist()) - {-1}
    )
    assert mask.kv_num_blocks[0, 0, 1] == len(
        set(mask.kv_indices[0, 0, 1].tolist()) - {-1}
    )


def test_block_deduplication() -> None:
    """Multiple point-to-point connections between blocks should result in a single block connection."""
    points = np.zeros((4, 2), dtype=np.float32)
    tree = KDTree(points)
    mask = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=4, k=4, block_size=2
    )

    assert mask.kv_num_blocks[0, 0, 0] == 2

    indices = mask.kv_indices[0, 0, 0, :2]
    assert torch.equal(indices.sort()[0], torch.tensor([0, 1], dtype=torch.int32))


def test_neighbor_coverage(simple_points: PointsFixture) -> None:
    """Every true nearest neighbor found by the KDTree should be contained within the active blocks of the mask."""
    tree, points = simple_points
    n_points = len(points)
    block_size = 2
    k = 2

    mask = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=n_points, k=k, block_size=block_size
    )
    _, neighbor_indices = tree.query(points, k=k)
    neighbor_indices = np.asarray(neighbor_indices)
    kv_indices = mask.kv_indices[0, 0]  # (num_blocks, max_neighbors)

    for q_idx in range(n_points):
        q_block = q_idx // block_size

        allowed_kv_blocks = set(kv_indices[q_block].tolist())
        allowed_kv_blocks.discard(-1)  # remove padding

        for neighbor_idx in neighbor_indices[q_idx]:
            target_kv_block = neighbor_idx // block_size
            assert target_kv_block in allowed_kv_blocks, (
                f"Point {q_idx} (Block {q_block}) has neighbor {neighbor_idx} "
                f"(Block {target_kv_block}), but mask only allows blocks {allowed_kv_blocks}"
            )


def test_symmetric_block_mask() -> None:
    """Test that the symmetric argument correctly mirrors the block adjacency matrix."""
    # Point 0 is isolated, Points 1, 2, 3 are clustered together.
    points = np.array(
        [
            [0.0, 0.0],  # Block 0
            [2.0, 0.0],  # Block 1
            [2.1, 0.0],  # Block 2
            [2.2, 0.0],  # Block 3
        ],
        dtype=np.float32,
    )
    tree = KDTree(points)
    block_size = 1
    k = 2  # Each point finds itself + its 1 closest neighbor

    mask_asym = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=4, k=k, block_size=block_size, symmetric=False
    )
    asym_indices = mask_asym.kv_indices[0, 0]

    assert 1 in asym_indices[0].tolist(), "Block 0 should attend to Block 1"
    assert 0 not in asym_indices[1].tolist(), (
        "Block 1 should NOT attend to Block 0 in asymmetric mode"
    )

    mask_sym = create_block_mask_from_kdtree(
        tree, points, n_points_unpadded=4, k=k, block_size=block_size, symmetric=True
    )
    sym_indices = mask_sym.kv_indices[0, 0]

    assert 1 in sym_indices[0].tolist(), "Block 0 should still attend to Block 1"
    assert 0 in sym_indices[1].tolist(), (
        "Block 1 MUST now attend to Block 0 because symmetric=True"
    )

    assert torch.equal(mask_sym.kv_indices, mask_sym.full_kv_indices), (
        "full_kv_indices must exactly match kv_indices when attend_all_mask_mod is used"
    )
