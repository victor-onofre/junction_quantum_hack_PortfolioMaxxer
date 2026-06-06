import numpy as np
import pytest
from galois import GF

from qualtran import QMontgomeryUInt
from qualtran.resource_counting import get_cost_value, QubitCount, QECGatesCost

from .berlekamp_massey_lfsr_modp import (
    BerlekampMasseyLFSR,
    InPlaceModMul,
    _berlekamp_massey_lfsr_large,
)


def assert_valid_recurrence(s, c):
    GFM = type(s[0])
    for i in range(len(c) + 1, len(s)):
        eval_si = sum([c[j] * s[i - j - 1] for j in range(len(c))], start=GFM(0))
        assert s[i] == eval_si


def generate_sequence(first_k_elements, n_elements, c) -> np.ndarray:
    assert len(first_k_elements) == len(c)
    GFM = type(first_k_elements[0])
    s = [*first_k_elements]
    for i in range(len(first_k_elements), n_elements):
        next_element = sum([c[j] * s[i - j - 1] for j in range(len(c))], start=GFM(0))
        s.append(next_element)
    return GFM(s)


@pytest.mark.parametrize("p", [3, 5, 7, 11])
def test_in_place_mod_mul(p: int):
    dtype = QMontgomeryUInt(bitsize=p.bit_length() + 1, modulus=p)
    bloq = InPlaceModMul(dtype)
    for x in range(1, p):
        for y in range(1, p):
            x_m, y_m = dtype.uint_to_montgomery(x), dtype.uint_to_montgomery(y)
            x_m_inv = dtype.montgomery_inverse(x_m)
            x_m_out, x_m_inv_out, y_m_out = bloq.call_classically(
                x=x_m, x_inv=x_m_inv, y=y_m
            )
            y_out = dtype.montgomery_to_uint(y_m_out)
            assert (x_m_out, x_m_inv_out, y_out) == (x_m, x_m_inv, (x * y) % p)


_SEQUENCES_TO_TEST = [
    ([2, 3], [2, -1]),
    ([1, 8], [1, 2]),
    ([1, 3, 5], [3, -1, -1]),
    ([0, 1], [1, 1]),
    ([0, 0, 2, 4], [2, 0, -2, 1]),
    ([0, 1, 4], [3, -3, 1]),
    ([0, 0, 0, 0, 0, 1], [1, 2, 3, 4, 5, 6]),
]


@pytest.mark.parametrize("s_init, expected_c", _SEQUENCES_TO_TEST)
@pytest.mark.parametrize("p", [29, 41, 43, 61, 103, 107, 241, 251, 277, 293])
def test_berlekamp_massey_lfs_modp_classical_gen_sequence(s_init, expected_c, p):
    if max(s_init) > GF(p).elements[-1]:
        return
    GFM = GF(p)
    expected_c = GFM([x % p for x in expected_c])
    s = generate_sequence(GFM(s_init), 5 * len(s_init), expected_c)
    assert_valid_recurrence(s, expected_c)
    max_c_len = min(len(s), 2 * len(expected_c))
    # Test using bloq
    dtype = QMontgomeryUInt(bitsize=p.bit_length() + 1, modulus=p)
    bloq = BerlekampMasseyLFSR(dtype, len(s), max_c_len)
    s_montgomery = np.array([dtype.uint_to_montgomery(int(x)) for x in s])
    c_montgomery = bloq.call_classically(s=s_montgomery)[1]
    c_out = GFM([dtype.montgomery_to_uint(int(x)) for x in c_montgomery])
    assert_valid_recurrence(s, c_out[1:])


@pytest.mark.parametrize(
    "n, max_len, p, toffoli, clifford, qubits",
    [
        (32, 16, 67, 999_850, 5_153_696, 907),
        (64, 32, 131, 4_607_692, 23_798_358, 1912),
        (128, 64, 257, 21_546_662, 110_333_236, 4101),
        (256, 128, 521, 101_011_904, 521_039_438, 8850),
        pytest.param(
            512, 256, 1031, 471_606_346, 2_444_572_208, 19103, marks=pytest.mark.slow
        ),
        pytest.param(
            1024,
            512,
            2153,
            2_186_280_548,
            11_380_033_666,
            41132,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_berlekamp_massey_lfsr_modp_paper_table_counts(
    n, max_len, p, toffoli, clifford, qubits
):
    bloq = BerlekampMasseyLFSR.build(p, n, max_len)
    gate_counts = get_cost_value(bloq, QECGatesCost())
    assert gate_counts.total_toffoli_only() == toffoli
    assert gate_counts.clifford == clifford
    assert get_cost_value(bloq, QubitCount()) == qubits


def test_berlekamp_massey_lfsr_modp_large_bloq_example_counts():
    bloq = _berlekamp_massey_lfsr_large.make()
    assert bloq.t_complexity().t == 86_939_288
    assert get_cost_value(bloq, QubitCount()) == 4_101
