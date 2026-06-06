import pytest
from dqi.bitwise import find_subspace_evasive_set, contains_affine_subspace
import pytest


@pytest.mark.parametrize("n,w", [(3, 2), (4, 2), (4, 3), (5, 2), (5, 3)])
def test_find_subspace_evasive_set_small_cases(n: int, w: int) -> None:
    result = find_subspace_evasive_set(n, w, 20, 0)
    assert not contains_affine_subspace(result, w)
