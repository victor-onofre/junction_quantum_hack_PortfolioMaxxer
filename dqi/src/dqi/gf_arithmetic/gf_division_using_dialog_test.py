import pytest
from galois import GF
from qualtran import QGF
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount
from dqi.gf_arithmetic.gf_division_using_dialog import GF2DivisionUsingDialog


@pytest.mark.parametrize('m', [2, 3, 4])
def test_gf2_division_using_dialog(m: int):
    gf = GF(2, m)
    bloq = GF2DivisionUsingDialog(QGF(2, m))
    bloq_adj = bloq.adjoint()
    for x in gf.elements[1:]:
        for y in gf.elements[1:]:
            junk, x_by_y = bloq.call_classically(x=x, y=y)
            assert x_by_y == x // y
            assert bloq_adj.call_classically(junk=junk, x_by_y=x_by_y) == (x, y)


@pytest.mark.parametrize(
    'm, n_qubits, n_toffoli',
    [
        (2, 22, 72),
        (3, 32, 146),
        (4, 38, 219),
        (5, 44, 306),
        (6, 50, 407),
        (7, 60, 567),
        (8, 66, 702),
        (9, 72, 851),
        (10, 78, 1014),
        (11, 84, 1191),
        (12, 90, 1382),
        (20, 142, 3_537),
        (50, 326, 19_620),
        (100, 630, 74_823),
    ],
)
def test_gf2_division_using_dialog_resource_counts(m: int, n_qubits: int, n_toffoli: int):
    bloq = GF2DivisionUsingDialog(QGF(2, m))
    assert get_cost_value(bloq, QubitCount()) == n_qubits
    assert get_cost_value(bloq, QECGatesCost()).total_toffoli_only() == n_toffoli
