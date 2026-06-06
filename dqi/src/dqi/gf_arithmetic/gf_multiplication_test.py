import itertools

import pytest
import qualtran.testing as qlt_testing
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join

from dqi.gf_arithmetic.gf_multiplication import GF2MulViaOptimizedPolyMul, PolyMul


@pytest.mark.parametrize('n', range(1, 10))
def test_binary_mult_decomposition(n):
    blq = PolyMul(n)
    qlt_testing.assert_valid_bloq_decomposition(blq)


@pytest.mark.parametrize('n', range(1, 10))
def test_binary_mult_bloq_counts(n):
    blq = PolyMul(n)
    qlt_testing.assert_equivalent_bloq_counts(
        blq, generalizer=(ignore_split_join, ignore_alloc_free)
    )


@pytest.mark.parametrize('n', range(1, 4))
def test_binary_mult_classical_action(n):
    blq = PolyMul(n)
    fg_polys = tuple(itertools.product(range(2), repeat=n))[1:]
    h_polys = [[0] * blq.signature[-1].shape[0]]

    qlt_testing.assert_consistent_classical_action(blq, f=fg_polys, g=fg_polys, h=h_polys)


@pytest.mark.parametrize('m_x', [[2, 1, 0], [3, 1, 0], [5, 2, 0], [8, 4, 3, 1, 0]])
def test_gf2mulmod_decomposition(m_x):
    blq = GF2MulViaOptimizedPolyMul(m_x)
    qlt_testing.assert_valid_bloq_decomposition(blq)


@pytest.mark.parametrize('m_x', [[2, 1, 0], [3, 1, 0], [5, 2, 0], [8, 4, 3, 1, 0]])
def test_gf2mulmod_bloq_counts(m_x):
    blq = GF2MulViaOptimizedPolyMul(m_x)
    qlt_testing.assert_equivalent_bloq_counts(
        blq, generalizer=(ignore_split_join, ignore_alloc_free)
    )


def test_gf2mulmod_classical_action():
    m_x = [3, 1, 0]
    blq = GF2MulViaOptimizedPolyMul(m_x)
    qlt_testing.assert_consistent_classical_action(blq, x=blq.gf.elements, y=blq.gf.elements)
