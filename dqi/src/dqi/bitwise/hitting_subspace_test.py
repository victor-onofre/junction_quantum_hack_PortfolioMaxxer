from dqi.bitwise import find_hitting_subspace


def test_find_hitting_subspace_simple():
    S = [(0, 0), (1, 0)]
    basis = find_hitting_subspace(S, 1)
    assert basis in ([(0, 1)], [(1, 1)])


def test_find_hitting_subspace_none():
    S = [(0, 0)]
    assert find_hitting_subspace(S, 1) is None
