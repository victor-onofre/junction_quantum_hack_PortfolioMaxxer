import pytest
from dqi.bitwise import find_max_overlap_affine_subspace, find_all_max_overlap_affine_subspaces


def test_find_max_overlap_affine_subspace():
    # n=3, S = {(0,0,0), (1,0,0), (0,1,0), (1,1,0)}
    S = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    # k=1 (lines)
    # Max overlap should be 2.
    # Line {(0,0,0), (1,0,0)} has overlap 2.
    # Line {(0,1,0), (1,1,0)} has overlap 2.
    # Line {(0,0,0), (0,1,0)} has overlap 2.
    # Line {(1,0,0), (1,1,0)} has overlap 2.
    max_overlap = find_max_overlap_affine_subspace(S, 1)
    assert max_overlap == 2

    # k=2 (planes)
    # Max overlap is 4, the plane {(x,y,0)} contains S.
    max_overlap = find_max_overlap_affine_subspace(S, 2)
    assert max_overlap == 4


def test_find_all_max_overlap_affine_subspaces():
    # n=3, S = {(0,0,0), (1,0,0), (0,1,0), (1,1,0)}
    S = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]

    # k=1 (lines)
    max_subspaces = find_all_max_overlap_affine_subspaces(S, 1)

    # There should be 6 lines with overlap 2.
    assert len(max_subspaces) == 6

    expected_subspaces = [
        {(0, 0, 0), (1, 0, 0)},
        {(0, 1, 0), (1, 1, 0)},
        {(0, 0, 0), (0, 1, 0)},
        {(1, 0, 0), (1, 1, 0)},
        {(0, 0, 0), (1, 1, 0)},
        {(1, 0, 0), (0, 1, 0)},
    ]

    # Convert the list of sets to a set of frozensets for comparison
    result_set = {frozenset(s) for s in max_subspaces}
    expected_set = {frozenset(s) for s in expected_subspaces}

    assert result_set == expected_set

    # k=2 (planes)
    max_subspaces_k2 = find_all_max_overlap_affine_subspaces(S, 2)
    assert len(max_subspaces_k2) == 1

    plane = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)}
    assert max_subspaces_k2[0] == plane


def test_find_all_max_overlap_empty_set():
    result = find_all_max_overlap_affine_subspaces([], 1)
    assert result == []


def test_find_max_overlap_empty_set():
    result = find_max_overlap_affine_subspace([], 1)
    assert result == 0
