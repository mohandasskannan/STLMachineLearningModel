"""Regression checks for the Text2Shape NRRD orientation discovered in the audit."""

import importlib.util
from pathlib import Path

import nrrd
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("pilot", Path(__file__).with_name("run_pilot.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


@pytest.mark.parametrize("reverse_vertical", [False, True])
def test_rgba_storage_axes_become_world_xyz(tmp_path, reverse_vertical):
    expected = np.zeros((32, 32, 32), dtype=np.uint8)
    expected[2:8, 4:11, 15:18] = 1
    stored = expected.transpose(1, 2, 0)
    directions = np.array([[np.nan] * 3, [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    if reverse_vertical:
        stored = np.flip(stored, axis=0)
        directions[1, 1] = -1
    rgba = np.stack([np.zeros_like(stored)] * 3 + [stored * 255])
    path = tmp_path / "shape.nrrd"
    nrrd.write(str(path), rgba, header={"space directions": directions}, index_order="F")
    actual, permutation = pilot.canonical_voxels(path)
    np.testing.assert_array_equal(actual, expected)
    assert permutation == [2, 0, 1]


def test_oblique_grid_is_rejected(tmp_path):
    grid = np.zeros((32, 32, 32), dtype=np.uint8)
    grid[3:6, 7:9, 10:12] = 1
    path = tmp_path / "oblique.nrrd"
    nrrd.write(str(path), grid, header={"space directions": [[1, 0.2, 0], [0, 1, 0], [0, 0, 1]]})
    with pytest.raises(ValueError, match="Oblique"):
        pilot.canonical_voxels(path)
