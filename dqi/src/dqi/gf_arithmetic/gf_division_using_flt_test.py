import pytest
from galois import GF
from qualtran import QGF
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount
from dqi.gf_arithmetic.gf_division_using_flt import GF2DivisionUsingFLT


@pytest.mark.parametrize('m', [3, 4, 5])
def test_gf2_division_using_flt(m: int):
    gf = GF(2, m)
    bloq = GF2DivisionUsingFLT(QGF(2, m))
    for x in gf.elements[1:]:
        for y in gf.elements[1:]:
            junk, x_by_y = bloq.call_classically(x=x, y=y)
            assert x_by_y == x // y


@pytest.mark.parametrize(
    'm, n_qubits, n_toffoli',
    [
        (2, 8, 3),
        (3, 15, 14),
        (4, 20, 27),
        (5, 30, 45),
        (6, 36, 72),
        (7, 42, 96),
        (8, 56, 135),
        (9, 63, 140),
        (10, 70, 195),
        (11, 77, 235),
        (12, 96, 306),
        (20, 181, 1071),
        (50, 501, 4920),
        (100, 1201, 18450),
    ],
)
def test_gf2_division_using_flt_resource_counts(m: int, n_qubits: int, n_toffoli: int):
    bloq = GF2DivisionUsingFLT(QGF(2, m))
    assert get_cost_value(bloq, QubitCount()) == n_qubits
    assert get_cost_value(bloq, QECGatesCost()).total_toffoli_only() == n_toffoli
