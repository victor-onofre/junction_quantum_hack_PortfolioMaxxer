import pytest
import math
from dqi.sparse_dicke_states.rank_to_combination import (
    rank_to_combination,
    rank_to_combination_divide_and_conquer,
)


@pytest.mark.parametrize('n, k', [(n, k) for n in range(1, 8) for k in range(0, n + 1)])
def test_rank_to_combination(n: int, k: int):
    n_choose_k = math.comb(n, k)
    res = set()
    for r in range(n_choose_k):
        comb = rank_to_combination(n, k, r)
        assert len(set(comb)) == len(comb) == k
        res.add(tuple(sorted(comb)))
    assert len(res) == n_choose_k


@pytest.mark.parametrize('n, k', [(n, k) for n in range(1, 8) for k in range(0, n + 1)])
def test_rank_to_combination_divide_and_conquer(n: int, k: int):
    n_choose_k = math.comb(n, k)
    res = set()
    for r in range(n_choose_k):
        comb = rank_to_combination_divide_and_conquer(n, k, r)
        assert len(set(comb)) == len(comb) == k
        res.add(tuple(sorted(comb)))
    assert len(res) == n_choose_k
