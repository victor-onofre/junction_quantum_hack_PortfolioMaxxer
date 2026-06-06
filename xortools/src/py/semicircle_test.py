"""Executes numerical checks for DQI semicircle law as python unit tests.

Example invocation from repo root:

$ pytest src/py/semicircle_test.py -v -n 16

"""

import semicircle

from typing import Any, Callable, Dict, List, Set, Tuple

import itertools
import math
import random
import time

import numpy as np
import pytest
import sympy

np.set_printoptions(linewidth=400)


def assert_fp_functions_equal(
    p: int, f1: Callable[[int], Any], f2: Callable[[int], Any]
) -> None:
    for x in range(p):
        assert abs(f1(x) - f2(x)) < 1e-10, (x, f1(x), f2(x))


# Key: (p, r, m, n, max_ell)
FAST_MAX_LIN_SAT_DESIRED_COUNT: Dict[Tuple[int, int, int, int, int], List[int]] = {
    # p = 2
    (2, 1, 7, 6, 0): 10,
    (2, 1, 7, 6, 1): 10,
    (2, 1, 7, 6, 2): 10,
    (2, 1, 10, 5, 0): 10,
    (2, 1, 10, 5, 1): 10,
    # p = 3
    (3, 1, 8, 6, 0): 2,
    (3, 1, 8, 6, 1): 2,
    (3, 1, 8, 6, 2): 2,
    (3, 2, 8, 6, 0): 2,
    (3, 2, 8, 6, 1): 2,
    (3, 2, 8, 6, 2): 2,
    # p = 5
    (5, 1, 5, 4, 1): 2,
    (5, 2, 5, 4, 1): 2,
    (5, 3, 5, 4, 1): 2,
    (5, 4, 5, 4, 1): 2,
    # p = 7
    (7, 3, 3, 2, 0): 1,
}
SLOW_MAX_LIN_SAT_DESIRED_COUNT: Dict[Tuple[int, int, int, int, int], List[int]] = {
    # p = 3
    (3, 1, 10, 9, 1): 1,
    (3, 1, 10, 9, 2): 1,
    (3, 1, 10, 9, 3): 1,
    (3, 2, 10, 9, 1): 1,
    (3, 2, 10, 9, 2): 1,
    (3, 2, 10, 9, 3): 1,
    # p = 5
    (5, 1, 7, 6, 1): 1,
    (5, 1, 7, 6, 2): 1,
    (5, 2, 7, 6, 1): 1,
    (5, 2, 7, 6, 2): 1,
    (5, 3, 7, 6, 1): 1,
    (5, 3, 7, 6, 2): 1,
    (5, 4, 7, 6, 1): 1,
    (5, 4, 7, 6, 2): 1,
    # p = 7
    (7, 3, 5, 4, 1): 1,
    # p = 19
    (19, 4, 4, 3, 1): 1,
}
FAST_MAX_LIN_SAT_INSTANCES: Dict[
    Tuple[int, int, int, int, int], Set[semicircle.MaxLinSat]
] = {}
SLOW_MAX_LIN_SAT_INSTANCES: Dict[
    Tuple[int, int, int, int, int], Set[semicircle.MaxLinSat]
] = {}


def prepare_test_data(
    counts: Dict[Tuple[int, int, int, int, int], List[int]],
    output: Dict[Tuple[int, int, int, int, int], Set[semicircle.MaxLinSat]],
    max_iter: int = 100000,
) -> None:
    for k in counts:
        output[k] = set()
    i = 0
    t0 = time.time()
    while counts:
        key1 = next(iter(counts.keys()))
        p, r, m, n, max_ell = key1
        mls = semicircle.MaxLinSat.random_instance(p, r, m, n)
        key2 = p, r, m, n, mls.max_ell()
        print(time.time() - t0, f"Found {key2} while looking for {key1}")
        if key2 in counts:
            output[key2].add(mls)
            counts[key2] -= 1
        if key2 in counts and counts[key2] == 0:
            del counts[key2]
        i += 1
        assert (
            i < max_iter
        ), "Could not find sufficiently many codes with requested parameters"


prepare_test_data(FAST_MAX_LIN_SAT_DESIRED_COUNT, FAST_MAX_LIN_SAT_INSTANCES)
prepare_test_data(SLOW_MAX_LIN_SAT_DESIRED_COUNT, SLOW_MAX_LIN_SAT_INSTANCES)
FAST_FLATTENED_MAX_LIN_SAT_INSTANCES = sum(
    (list(batch) for batch in FAST_MAX_LIN_SAT_INSTANCES.values()), []
)
SLOW_FLATTENED_MAX_LIN_SAT_INSTANCES = sum(
    (list(batch) for batch in SLOW_MAX_LIN_SAT_INSTANCES.values()), []
)
ALL_FLATTENED_MAX_LIN_SAT_INSTANCES = (
    FAST_FLATTENED_MAX_LIN_SAT_INSTANCES + SLOW_FLATTENED_MAX_LIN_SAT_INSTANCES
)

assert len(FAST_FLATTENED_MAX_LIN_SAT_INSTANCES) == 71
assert len(SLOW_FLATTENED_MAX_LIN_SAT_INSTANCES) == 16


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_g_i(mls):
    """Verify explicit formula for g_i."""
    for i in range(1, mls.m + 1):
        Fi = mls.get_fi(i)

        def expected_g_i(x: int) -> float:
            if x in Fi:
                return 2 - 2 * mls.r / mls.p
            return -2 * mls.r / mls.p

        actual_g_i = mls.make_g_i(i)
        assert_fp_functions_equal(mls.p, expected_g_i, actual_g_i)


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_g_i_tilde(mls):
    r"""Verify explicit formula for \tilde{g}_i."""
    for i in range(1, mls.m + 1):
        Fi = mls.get_fi(i)

        def expected_g_i_tilde(y: int) -> float:
            if y == 0:
                return 0
            return (
                2 * sum(np.exp(2j * np.pi * v * y / mls.p) for v in Fi) / np.sqrt(mls.p)
            )

        actual_g_i_tilde = mls.make_g_i_tilde(i)
        assert_fp_functions_equal(mls.p, expected_g_i_tilde, actual_g_i_tilde)


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_h_i(mls):
    """Verify explicit formula for h_i."""
    for i in range(1, mls.m + 1):
        Fi = mls.get_fi(i)

        def expected_h_i(x: int) -> float:
            if x in Fi:
                return np.sqrt((mls.p - mls.r) / mls.r / mls.p)
            return -np.sqrt(mls.r / mls.p / (mls.p - mls.r))

        actual_h_i = mls.make_h_i(i)
        assert_fp_functions_equal(mls.p, expected_h_i, actual_h_i)


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_h_i_tilde(mls):
    r"""Verify explicit formula for \tilde{g}_i."""
    for i in range(1, mls.m + 1):
        Fi = mls.get_fi(i)

        def expected_h_i_tilde(y: int) -> float:
            if y == 0:
                return 0
            return sum(np.exp(2j * np.pi * v * y / mls.p) for v in Fi) / np.sqrt(
                mls.r * (mls.p - mls.r)
            )

        actual_h_i_tilde = mls.make_h_i_tilde(i)
        assert_fp_functions_equal(mls.p, expected_h_i_tilde, actual_h_i_tilde)


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_g_i_norm(mls):
    r"""Verify formula for the norm of the p-vector [g_i(x) for x in \mathbb{F}_p]."""
    v = np.zeros(mls.p, dtype=complex)
    for i in range(1, mls.m + 1):
        g_i = mls.make_g_i(i)
        for x in range(mls.p):
            v[x] = g_i(x)
        assert (
            abs(np.linalg.norm(v) - 2 * np.sqrt(mls.r * (mls.p - mls.r) / mls.p))
            < 1e-10
        )


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_g_i_tilde_norm(mls):
    r"""Verify formula for the norm of the p-vector [\tilde{g}_i(x) for x in \mathbb{F}_p]."""
    v = np.zeros(mls.p, dtype=complex)
    for i in range(1, mls.m + 1):
        g_i_tilde = mls.make_g_i_tilde(i)
        for x in range(mls.p):
            v[x] = g_i_tilde(x)
        assert (
            abs(np.linalg.norm(v) - 2 * np.sqrt(mls.r * (mls.p - mls.r) / mls.p))
            < 1e-10
        )


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_h_i_norm(mls):
    r"""Verify formula for the norm of the p-vector [h_i(x) for x in \mathbb{F}_p]."""
    v = np.zeros(mls.p, dtype=complex)
    for i in range(1, mls.m + 1):
        h_i = mls.make_h_i(i)
        for x in range(mls.p):
            v[x] = h_i(x)
        assert abs(np.linalg.norm(v) - 1) < 1e-10


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_h_i_tilde_norm(mls):
    r"""Verify formula for the norm of the p-vector [\tilde{h}_i(x) for x in \mathbb{F}_p]."""
    v = np.zeros(mls.p, dtype=complex)
    for i in range(1, mls.m + 1):
        h_i_tilde = mls.make_h_i_tilde(i)
        for x in range(mls.p):
            v[x] = h_i_tilde(x)
        assert abs(np.linalg.norm(v) - 1) < 1e-10


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_phi_independent_of_i(mls):
    for i in range(1, mls.m + 1):
        phi = mls.make_phi_i(i)
        s = sum(phi(z) for z in range(1, mls.p))
        assert abs(s - mls.r * (mls.p - mls.r)) < 1e-10


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_convolution_expression(mls):
    """Verify convolution expression that appears in the formula for diagonal elements."""
    om = semicircle.omega(mls.p)
    for i in range(1, mls.m + 1):
        for v in mls.get_fi(i):
            lhs = 0
            h_i_tilde = mls.make_h_i_tilde(i)
            for a in range(mls.p):
                for z in range(mls.p):
                    lhs += (om ** (-a * v)) * h_i_tilde(a - z) * h_i_tilde(z)
            rhs = (mls.p - mls.r) / mls.r
            assert abs(lhs - rhs) < 1e-10, (lhs, rhs, lhs - rhs)


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_lemma_4_1(mls):
    r"""Verify formula for sum of \tilde{h} over symbol strings of a fixed Hamming weight."""
    h_tilde = mls.make_h_tilde()
    for k in range(mls.m):
        actual_sum = 0
        for s in semicircle.generate_symbol_strings_of_fixed_hamming_weight(
            mls.p, mls.m, k
        ):
            y = np.array(s)
            actual_sum += abs(h_tilde(y)) ** 2
        expected_sum = math.comb(mls.m, k)
        assert abs(actual_sum - expected_sum) < 1e-10, (
            actual_sum,
            expected_sum,
            actual_sum - expected_sum,
        )


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_corollary_4_1(mls):
    r"""Verify normalization condition for Fourier-transformed DQI states given in Corollary 4.1."""
    for ell in range(mls.max_ell()):
        w = np.random.rand(ell + 1)
        psi = mls.make_fdqi_state(w)
        w_sql2 = w.T.conj() @ w
        psi_sql2 = psi.T.conj() @ psi
        assert abs(w_sql2 - psi_sql2) < 1e-10, (w_sql2, psi_sql2, w_sql2 - psi_sql2)


@pytest.mark.parametrize(
    "m, ell, d",
    [
        (m, ell, d)
        for (m, ell), d in itertools.product(
            [(10000, 1000), (40000, 1000), (40000, 2000)],
            [0, 1, 5, 20],
        )
    ],
)
def test_lemma_4_2(m, ell, d):
    r"""Verify asymptotic formula for maximum eigenvalue of tridiagonal matrix A divided by m."""
    mat = semicircle.make_matrix_A(m, ell, d) / m
    eigs, _ = np.linalg.eigh(mat)
    actual_max_eigenvalue = max(eigs)
    expected_max_eigenvalue = d * ell / m + 2 * np.sqrt(ell * (m - ell)) / m
    assert abs(actual_max_eigenvalue - expected_max_eigenvalue) < 0.025, (
        actual_max_eigenvalue,
        expected_max_eigenvalue,
        actual_max_eigenvalue - expected_max_eigenvalue,
    )


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_constraints_satisfied(mls):
    """Verify complex expression for s(x)."""

    def actual_s(x: np.ndarray) -> complex:
        sx = 0
        om = semicircle.omega(mls.p)
        for i in range(1, mls.m + 1):
            Fi = mls.get_fi(i)
            for v in Fi:
                for a in range(mls.p):
                    sx += om ** (a * ((mls.get_bi(i) @ x) - v))
        return sx / mls.p

    expected_s = mls.num_constraints_satisfied

    for _ in range(100):
        x = np.random.randint(low=0, high=mls.p, size=(mls.n,))
        assert abs(actual_s(x) - expected_s(x)) < 1e-10


@pytest.mark.parametrize("mls", ALL_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_innermost_sum_is_inverse_fourier_transform(mls):
    y = np.random.randint(low=0, high=mls.p, size=(mls.m,))

    rhs = 0
    om = semicircle.omega(mls.p)
    for i in range(1, mls.m + 1):
        if y[i - 1] != 0:
            continue
        h_i_tilde = mls.make_h_i_tilde(i)
        for v in mls.get_fi(i):
            for a in range(1, mls.p):
                rhs += h_i_tilde(a) * (om ** (-a * v))

    k = semicircle.hamming_weight(y)
    lhs = (mls.m - k) * np.sqrt(mls.r * (mls.p - mls.r))

    assert abs(rhs - lhs) < 1e-10, (rhs, lhs)


@pytest.mark.parametrize("p", (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 101))
def test_fourier_clock_and_shift(p):
    r"""Verify that FZF^\dagger == X^{-1}."""
    f = semicircle.make_single_qudit_fourier_matrix(p)
    z = semicircle.make_single_qudit_clock_matrix(p)
    x = semicircle.make_single_qudit_shift_matrix(p)
    assert f.shape == (p, p)
    assert z.shape == (p, p)
    assert x.shape == (p, p)
    assert np.all(x @ x.T == np.eye(p))
    assert sum(sum(np.abs(z @ z.conj() - np.eye(p)))) < 1e-10
    assert sum(sum(abs(f @ z @ f.T.conj() - x.T))) < 1e-10


@pytest.mark.parametrize("mls", FAST_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_sf_observable_spectrum(mls):
    s = mls.make_sf_observable()
    eigenvalues, _ = np.linalg.eig(s)
    assert 0 <= min(eigenvalues)
    assert max(eigenvalues) <= mls.m


@pytest.mark.parametrize("mls", FAST_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_s_in_terms_of_z(mls):
    S1 = mls.make_sf_observable()

    S2 = np.zeros((mls.p**mls.n, mls.p**mls.n), dtype=complex)
    om = semicircle.omega(mls.p)
    for i in range(1, mls.m + 1):
        for v in mls.get_fi(i):
            for a in range(mls.p):
                S2 += om ** (-a * v) * mls.make_z_product(i, a)
    S2 = S2 / mls.p

    assert sum(sum(abs(S1 - S2))) < 1e-10


@pytest.mark.parametrize("mls", FAST_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_s_in_terms_of_x(mls):
    S = mls.make_sf_observable()
    F1 = semicircle.make_single_qudit_fourier_matrix(mls.p)
    F = semicircle.kron_pow(F1, mls.n)
    Fd = F.T.conj()
    S1 = F @ S @ Fd

    S2 = np.zeros((mls.p**mls.n, mls.p**mls.n), dtype=complex)
    om = semicircle.omega(mls.p)
    for i in range(1, mls.m + 1):
        for v in mls.get_fi(i):
            for a in range(mls.p):
                S2 += om ** (-a * v) * mls.make_x_product(i, a)
    S2 = S2 / mls.p

    assert sum(sum(abs(S1 - S2))) < 1e-8


@pytest.mark.parametrize("mls", FAST_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_action_of_x_on_yB(mls):
    y = np.random.randint(low=0, high=mls.p, size=(mls.m,))
    e = np.eye(mls.m, dtype=int)
    for i in range(1, mls.m + 1):
        for a in range(mls.p):
            s1 = (y @ mls.B) % mls.p
            s2 = ((y - a * e[i - 1]) @ mls.B) % mls.p
            i1 = semicircle.symbol_string_to_index(mls.p, s1)
            i2 = semicircle.symbol_string_to_index(mls.p, s2)
            y1 = np.zeros(mls.p**mls.n, dtype=complex)
            y2 = np.zeros(mls.p**mls.n, dtype=complex)
            y1[i1] = 1
            y2[i2] = 1

            X = mls.make_x_product(i, a)
            y3 = X @ y1
            assert y3.T @ y3 == 1
            assert min(y3) == 0
            assert max(y3) == 1
            assert sum(abs(y2 - y3)) < 1e-10


def chase_quadratic_form(mls: semicircle.MaxLinSat, ell: int) -> None:
    """Chases our quadratic form through the proof of theorem 2.1."""
    # Check the inputs are valid.
    assert 2 * ell + 1 < mls.dmin, (ell, 2 * ell + 1, mls.dmin)

    # Prepare some useful scalars, vectors, functions, and operators.
    A = semicircle.make_matrix_A(
        mls.m, ell, (mls.p - 2 * mls.r) / np.sqrt(mls.r * (mls.p - mls.r))
    )
    D = np.diag(range(ell + 1))
    om = semicircle.omega(mls.p)
    e = np.eye(mls.m, dtype=int)
    F1 = semicircle.make_single_qudit_fourier_matrix(mls.p)
    F = semicircle.kron_pow(F1, mls.n)
    h_tilde = mls.make_h_tilde()
    N1 = mls.make_sf_observable()

    # Prepare a DQI state.
    w = np.random.rand(ell + 1)
    w = w / np.sqrt(w.T.conj() @ w)
    psi = mls.make_fdqi_state(w)
    assert abs(psi.T.conj() @ psi - 1) < 1e-10

    # Starting point: quadratic form involving a DQI state and the observable S_f whose eigenvalues are the numbers of satisfied constraints.
    S1 = F @ N1 @ F.T.conj()
    q1 = psi.T.conj() @ S1 @ psi
    print(f"{q1=}", q1 < mls.m)
    assert 0 <= q1 <= mls.m

    # Next up: quadratic form with the initial expression for S_f in terms of the clock operator.
    N2 = np.zeros((mls.p**mls.n, mls.p**mls.n), dtype=complex)
    for i in range(1, mls.m + 1):
        for v in mls.get_fi(i):
            for a in range(mls.p):
                N2 += om ** (-a * v) * mls.make_z_product(i, a)
    N2 = N2 / mls.p
    S2 = F @ N2 @ F.T.conj()
    q2 = psi.T.conj() @ S2 @ psi
    print(f"{q2=}", q2 < mls.m)
    assert abs(q2 - q1) < 1e-10

    # Then in terms of the shift operator following QFT.
    S3 = np.zeros((mls.p**mls.n, mls.p**mls.n), dtype=complex)
    for i in range(1, mls.m + 1):
        for v in mls.get_fi(i):
            for a in range(mls.p):
                S3 += om ** (-a * v) * mls.make_x_product(i, a)
    S3 = S3 / mls.p
    q3 = psi.T.conj() @ S3 @ psi
    print(f"{q3=}", q3 < mls.m)
    assert abs(q3 - q1) < 1e-10

    # And then in terms of shift operator's effect on computational basis.
    q5 = 0
    cnt = 0
    for k1 in range(ell + 1):
        for k2 in range(ell + 1):
            inner_sum = 0
            for s1 in semicircle.generate_symbol_strings_of_fixed_hamming_weight(
                mls.p, mls.m, k1
            ):
                y1 = np.array(s1)
                for s2 in semicircle.generate_symbol_strings_of_fixed_hamming_weight(
                    mls.p, mls.m, k2
                ):
                    y2 = np.array(s2)
                    innermost_sum = 0
                    for i in range(1, mls.m + 1):
                        for v in mls.get_fi(i):
                            for a in range(mls.p):
                                t1 = (y1 @ mls.B) % mls.p
                                t2 = ((y2 - a * e[i - 1]) @ mls.B) % mls.p
                                i1 = semicircle.symbol_string_to_index(mls.p, t1)
                                i2 = semicircle.symbol_string_to_index(mls.p, t2)
                                yb1 = np.zeros(mls.p**mls.n, dtype=complex)
                                yb2 = np.zeros(mls.p**mls.n, dtype=complex)
                                yb1[i1] = 1
                                yb2[i2] = 1
                                val = (
                                    (om ** (-a * v))
                                    * (yb1.T @ yb2)
                                    * h_tilde(y1).conjugate()
                                    * h_tilde(y2)
                                )
                                innermost_sum += val
                    inner_sum += innermost_sum
            q5 += (
                w[k1].conj()
                * w[k2]
                * inner_sum
                / np.sqrt(math.comb(mls.m, k1) * math.comb(mls.m, k2))
            )
    q5 = q5 / mls.p
    print(f"{q5=}", q5 < mls.m)
    assert abs(q5 - q1) < 1e-10

    # Next, after breaking out into four cases based on Hamming weight.
    q6_sup = 0
    q6_sub = 0
    for k in range(ell):
        inner_sum = 0
        for s in semicircle.generate_symbol_strings_of_fixed_hamming_weight(
            mls.p, mls.m, k
        ):
            y = np.array(s)
            innermost_sum = 0
            for i in range(1, mls.m + 1):
                if y[i - 1] != 0:
                    continue
                h_i_tilde = mls.make_h_i_tilde(i)
                for v in mls.get_fi(i):
                    for a in range(1, mls.p):
                        innermost_sum += (om ** (-a * v)) * h_i_tilde(a)
            expected_innermost_sum = (mls.m - k) * np.sqrt(mls.r * (mls.p - mls.r))
            assert abs(innermost_sum - expected_innermost_sum) < 1e-10
            inner_sum += innermost_sum * abs(h_tilde(y)) ** 2
        assert (
            abs(inner_sum - (math.comb(mls.m, k) * expected_innermost_sum)) < 1e-10
        ), (abs(inner_sum - math.comb(mls.m, k)),)
        q6_sup += (
            w[k].conj()
            * w[k + 1]
            * inner_sum
            / np.sqrt(math.comb(mls.m, k) * math.comb(mls.m, k + 1))
        )
        q6_sub += (
            w[k + 1].conj()
            * w[k]
            * inner_sum
            / np.sqrt(math.comb(mls.m, k + 1) * math.comb(mls.m, k))
        )
    q6_diag = mls.m * mls.r * (abs(w[0]) ** 2)
    for k in range(1, ell + 1):
        q6_diag += (
            (mls.m * mls.r + k * (mls.p - 2 * mls.r))
            * math.comb(mls.m, k)
            * (abs(w[k]) ** 2)
            / math.comb(mls.m, k)
        )
    q6 = (q6_sup + q6_sub + q6_diag) / mls.p
    print(f"{q6=}", q6 < mls.m)
    assert abs(q6 - q1) < 1e-10

    # Finally, check the expression in terms of the tridiagonal matrix A.
    q9 = (
        mls.m * mls.r / mls.p
        + np.sqrt(mls.r * (mls.p - mls.r)) * (w.T.conj() @ A @ w) / mls.p
    )
    print(f"{q9=}", q9 < mls.m)
    assert abs(q9 - q1) < 1e-10


@pytest.mark.parametrize("mls", FAST_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_quadratic_form(mls):
    """Verify transformations of our quadratic form throughout the proof of theorem 2.1."""
    for ell in range(mls.max_ell() + 1):
        chase_quadratic_form(mls, ell)


def state_with_poly_amplitudes(mls, poly, normalize_cost_function) -> np.ndarray:
    """Returns state sum_x poly(f_1(b_1 x), ..., f_m(b_m x))|x>."""
    psi = np.zeros(mls.p ** mls.n, dtype=complex)
    for s in semicircle.generate_all_symbol_strings(mls.p, mls.n):
        x = np.array(s)
        values = []
        for i in range(1, mls.m + 1):
            if normalize_cost_function:
                f = mls.make_h_i(i)
            else:
                f = mls.make_f_i(i)
            b = mls.get_bi(i)
            values.append((i - 1, f((b @ x) % mls.p)))
        j = semicircle.symbol_string_to_index(mls.p, x)
        amplitude = complex(sympy.EX.to_sympy(poly.subs(values)))
        psi[j] = amplitude
    return psi


def check_u_vs_w(mls, normalize_cost_function):
    """Check that w_k obtained as inner product of quantum states and from u_k agree."""
    if normalize_cost_function:
        def w_by_u(p, m, n, k):
            return np.sqrt((p ** (n - k)) * math.comb(m, k))
    else:
        def w_by_u(p, m, n, k):
            return np.sqrt((p ** n) * math.comb(m, k))

    ALL_VARS = 'abcdefghijklmnopqrstuvxyz'
    poly_variables = ','.join(ALL_VARS[:mls.m])
    poly_ring = sympy.polys.rings.PolyRing(poly_variables, 'EX')
    for ell in range(mls.max_ell() + 1):
        # Compute elementary symmetric polynomials and a random polynomial P(f)
        poly = 0
        elementary_symmetric_polys = []
        u = np.random.rand(ell + 1)
        for k in range(ell + 1):
            pk = poly_ring.symmetric_poly(k)
            poly += u[k] * pk
            elementary_symmetric_polys.append(pk)

        # Compute DQI state
        dqi_state = state_with_poly_amplitudes(mls, poly, normalize_cost_function)
        norm = np.linalg.norm(dqi_state)
        dqi_state = dqi_state / norm
        poly = poly / norm
        u = u / norm

        # Compute quantum states corresponding to the elementary symmetric polynomials
        pk_states = []
        for k, elementary_symmetric_poly in enumerate(elementary_symmetric_polys):
            pk_state = state_with_poly_amplitudes(mls, elementary_symmetric_poly, normalize_cost_function)
            pk_state = pk_state / w_by_u(mls.p, mls.m, mls.n, k)
            pk_states.append(pk_state)

        # Verify the states form an orthonormal set
        M = np.zeros((ell + 1, ell + 1), dtype=complex)
        for k1 in range(ell + 1):
            for k2 in range(ell + 1):
                M[k1, k2] = pk_states[k1].T.conj() @ pk_states[k2]
        assert np.linalg.norm(M - np.eye(ell + 1)) < 1e-12

        # Verify the relationship between u_k and w_k
        for k, pk_state in enumerate(pk_states):
            actual_wk = dqi_state.T.conj() @ pk_state
            expected_wk = u[k] * w_by_u(mls.p, mls.m, mls.n, k)
            assert abs(actual_wk - expected_wk) < 1e-12


@pytest.mark.parametrize("mls", [mls for mls in FAST_FLATTENED_MAX_LIN_SAT_INSTANCES if mls.p == 2])
def test_u_vs_w_xorsat(mls):
    """Verify relationship between u and w in section 3.1.1."""
    check_u_vs_w(mls, normalize_cost_function=False)


@pytest.mark.parametrize("mls", FAST_FLATTENED_MAX_LIN_SAT_INSTANCES)
def test_u_vs_w_linsat(mls):
    """Verify relationship between u and w in section 3.2.1."""
    check_u_vs_w(mls, normalize_cost_function=True)
