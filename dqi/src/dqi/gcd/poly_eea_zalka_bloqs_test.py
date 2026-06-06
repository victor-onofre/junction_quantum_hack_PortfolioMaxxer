import pytest
import numpy as np
from qualtran import QGF
from galois import Poly, GF
from dqi.gcd.poly_eea_zalka_bloqs import (
    _PolyEEAZalkaImpl,
    _GetAddIf,
    _GetSubIf,
    MultiplyAndAddOrSub,
    StopEEA_LFSR,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount
from dqi.gcd.poly_eea_zalka_test import (
    _SEQUENCES_TO_TEST,
    generate_sequence,
    assert_valid_recurrence,
)
from qualtran.simulation.classical_sim import do_phased_classical_simulation


def _get_poly_eea_zalka_instance_small() -> _PolyEEAZalkaImpl:
    qgf = QGF(2, 3)
    m = 4
    n_steps = 5
    a = Poly([1, 2, 4, 7], field=qgf.gf_type)
    b = Poly([4, 3, 2, 1], field=qgf.gf_type)
    bloq = _PolyEEAZalkaImpl(qgf, m=m, n_steps=n_steps, stop_bloq=StopEEA_LFSR(m))
    return bloq, a, b


def test_swap_a_and_b():
    bloq, a, b = _get_poly_eea_zalka_instance_small()
    bloq = bloq.poly_eea_zalka_impl_step.swap_a_and_b
    kwargs = {
        'a_array': a.coefficients(size=4),
        'b_array': b.coefficients(size=4),
        'n_A': 3,
        'n_B': 4,
        'n_a': 1,
        'n_b': 0,
        'counter': 3,
        'flag': 1,
        'is_hc_zero': 1,
    }
    assert bloq.call_classically(**kwargs)[2] == 4
    assert bloq.decompose_bloq().call_classically(**kwargs)[2] == 4
    kwargs.pop('counter')
    assert bloq.call_classically(**kwargs, counter=2)[2] == 3

    expected_toffoli = bloq.m * bloq.qgf.bitsize + 2 * bloq.quint_logm.bitsize + 6
    assert get_cost_value(bloq, QECGatesCost()).total_toffoli_only() == expected_toffoli
    assert get_cost_value(bloq, QubitCount()) == bloq.signature.n_qubits() + 2


def test_get_add_if():
    bloq = _GetAddIf(QGF(2, 4), 16, i=5)
    assert bloq.call_classically(n_a=8, counter=[1, 1])[-1] == True
    assert bloq.call_classically(n_a=4, counter=[1, 1])[-1] == False

    bloq = _GetSubIf(QGF(2, 4), 16, i=5)
    assert bloq.call_classically(n_A=11, counter=[0, 0])[-1] == True
    assert bloq.call_classically(n_A=16, counter=[0, 0])[-1] == True
    assert bloq.call_classically(n_A=10, counter=[0, 0])[-1] == False
    assert bloq.call_classically(n_A=5, counter=[0, 0])[-1] == False


def test_multiply_and_add_or_sub():
    bloq = MultiplyAndAddOrSub(QGF(2, 4), m=7)
    rng = np.random.default_rng(42)
    cbloq = bloq.decompose_bloq()
    kwargs = {
        'a_array': GF(2, 4)([1, 1, 10, 0, 13, 12, 1]),
        'b_array': GF(2, 4)([0, 0, 0, 0, 4, 7, 3]),
        'n_A': 3,
        'n_a': 1,
        'n_q': 2,
        'counter': 0,
        'is_hc_zero': True,
    }
    bloq_out = bloq.call_classically(**kwargs)
    cbloq_out, phase = do_phased_classical_simulation(cbloq, kwargs, rng)
    assert phase == 1
    # cbloq_out = cbloq.call_classically(**kwargs)
    assert np.all(np.all(a == b) for a, b in zip(bloq_out, cbloq_out.values()))


@pytest.mark.parametrize('s, expected_c', _SEQUENCES_TO_TEST)
@pytest.mark.parametrize('m', range(3, 12))
def test_poly_eea_zalka_lfsr_sync_classical_gen_sequence_opt(s, expected_c, m: int):
    qgf = QGF(2, m)
    GFM = qgf.gf_type
    if max(s) > GFM.elements[-1]:
        pytest.skip()
    expected_c = GFM([abs(x) for x in expected_c])
    s = generate_sequence(GFM(s[: len(expected_c)]), len(s), expected_c)
    assert_valid_recurrence(s, expected_c)
    # c = poly_eea_zalka_lfsr_sync(s)
    m = len(s) + 5
    print(f'{m=}')
    bloq = _PolyEEAZalkaImpl(qgf=qgf, m=m, n_steps=3 * m + 6, stop_bloq=StopEEA_LFSR(m))
    A = Poly(s[::-1], field=GFM)
    B = Poly.Degrees(coeffs=[1], degrees=[len(s)], field=GFM)
    a_array, b_array = GFM.Zeros(m), GFM.Zeros(m)
    a_array[0] = 1
    a_array[m - A.degree - 1 :] = A.coeffs[::-1]
    b_array[m - B.degree - 1 :] = B.coeffs[::-1]
    n_A, n_B, n_a, n_b, n_q = A.degree + 1, B.degree + 1, 1, 0, 0
    flag, counter, halting_counter = 1, 0, 0

    bloq_out = bloq.call_classically(
        a_array=a_array,
        b_array=b_array,
        n_A=n_A,
        n_B=n_B,
        n_a=n_a,
        n_b=n_b,
        n_q=n_q,
        flag=flag,
        counter=counter,
        halting_counter=halting_counter,
    )

    b_array, n_b = bloq_out[1], bloq_out[5]
    b = Poly(b_array[:n_b], GFM)
    c = (b // (GFM(0) - b.coeffs[-1])).coeffs[::-1]
    # check output is correct.
    assert_valid_recurrence(s, c[1:])
