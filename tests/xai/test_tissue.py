import numpy as np
import pytest

from nuclei_graph.xai.tissue import TissueMask


@pytest.fixture
def mask() -> TissueMask:
    tissue = np.zeros((40, 60), dtype=bool)  # 2 µm/px: 80 x 120 µm
    tissue[10:30, 10:40] = True  # x 20-80 µm, y 20-60 µm
    return TissueMask(tissue, um_per_px=2.0)


def test_boundary_distance(mask: TissueMask) -> None:
    distance = mask.boundary_distance_um(
        np.array([50.0, 22.0, 100.0]), np.array([40.0, 40.0, 10.0])
    )
    assert distance[0] == pytest.approx(20.0, abs=2.0)  # centre: 20 µm to the top edge
    assert distance[1] == pytest.approx(2.0, abs=2.0)  # next to the left edge
    assert distance[2] == 0.0  # outside tissue


def test_cell_fraction(mask: TissueMask) -> None:
    fraction = mask.cell_fraction(cell_um=40.0)  # 3 x 2 cells
    assert fraction.shape == (3, 2)
    # tissue (x 20-80, y 20-60) covers half of cell (0, 0) in x and in y
    assert fraction[0, 0] == pytest.approx(0.25)
    assert fraction[1, 0] == pytest.approx(0.5)
    assert fraction[1, 1] == pytest.approx(0.5)
    assert fraction[2, 0] == pytest.approx(0.0)
    # total tissue area is preserved
    assert fraction.sum() * 40.0**2 == pytest.approx(60.0 * 40.0)


def test_cell_fraction_bounded_for_non_integer_cells() -> None:
    mask = TissueMask(np.ones((300, 300), dtype=bool), um_per_px=3.89)
    fraction = mask.cell_fraction(cell_um=250.0)
    assert fraction.max() <= 1.0
    assert fraction[:-1, :-1].min() == 1.0  # interior cells are fully tissue
    assert 0.0 < fraction[-1, -1] < 1.0  # the corner cell lies partly outside
