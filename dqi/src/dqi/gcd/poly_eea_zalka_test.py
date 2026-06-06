import numpy as np
import galois
import pytest
from galois import GF, Poly

from dqi.gcd.poly_eea_zalka import (
    poly_eea_zalka_gcd,
    poly_eea_zalka_lfsr,
    poly_eea_zalka_lfsr_sync,
    poly_eea_zalka_gcd_sync,
)

rs = np.random.default_rng(42)


_SEQUENCES_TO_TEST = [
    ([0, 2, 3, 4, 5, 6, 7, 8], [2, -1]),
    ([1, 8, 10, 26, 46], [1, 2]),
    ([1, 3, 5, 11, 25, 59, 141, 339], [3, -1, -1]),
    ([0, 1, 1, 2, 3, 5, 8, 13], [1, 1]),
    ([0, 0, 2, 4, 8, 12, 18, 24], [2, 0, -2, 1]),
    ([0, 1, 4, 9, 16, 25, 36], [3, -3, 1]),
]


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


def test_poly_eea_zalka_small():
    p = 7
    gf = GF(7)
    r0 = Poly(np.array([2, 7, 1, 8, 2, 8, 1, 8]) % p, field=gf)
    r1 = Poly(np.array([3, 1, 4, 1, 5, 9, 2]) % p, field=gf)
    g = poly_eea_zalka_gcd(r1, r0)
    assert g == galois.gcd(r0, r1)
    r = Poly(np.array([2, 3]), field=gf)
    r0, r1 = r0 * r, r1 * r
    g = poly_eea_zalka_gcd(r1, r0)
    assert g == galois.gcd(r0, r1)


@pytest.mark.parametrize(
    'p', [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**15, 2**20, 2**30, 2**40] + [7, 19, 31, 127]
)
def test_poly_eea_zalka(p: int):
    gf = GF(p)
    for _ in range(5):
        logp = int(np.log2(p))
        roots = gf.Random(logp, seed=rs)
        polys = np.array([Poly([1, r], field=gf) for r in roots])
        # Balanced A and B polynomials
        A = np.prod(polys[: logp // 2 + 2], initial=Poly([1], field=gf))
        B = np.prod(polys[logp // 2 :], initial=Poly([1], field=gf))
        expected_g = galois.gcd(A, B)
        g = poly_eea_zalka_gcd(A, B)
        assert g == expected_g
        # Skewed A and B polynomials
        A = np.prod(polys[: logp - 1], initial=Poly([1], field=gf))
        B = np.prod(polys[logp - 2 :], initial=Poly([1], field=gf))
        expected_g = galois.gcd(A, B)
        g = poly_eea_zalka_gcd(A, B)
        assert g == expected_g


@pytest.mark.parametrize(
    'p', [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**15, 2**20, 2**30, 2**40] + [7, 19, 31, 127]
)
def test_poly_eea_zalka_sync(p: int):
    gf = GF(p)
    n = int(np.log2(p))
    for _ in range(5):
        roots = gf.Random(n, seed=rs)
        polys = np.array([Poly([1, r], field=gf) for r in roots])
        # Balanced A and B polynomials
        A = np.prod(polys[: n // 2 + 2], initial=Poly([1], field=gf))
        B = np.prod(polys[n // 2 :], initial=Poly([1], field=gf))
        expected_g = galois.gcd(A, B)
        g = poly_eea_zalka_gcd_sync(A, B)
        assert g == expected_g
        # Skewed A and B polynomials
        A = np.prod(polys[: n - 1], initial=Poly([1], field=gf))
        B = np.prod(polys[n - 2 :], initial=Poly([1], field=gf))
        expected_g = galois.gcd(A, B)
        g = poly_eea_zalka_gcd_sync(A, B)
        assert g == expected_g


@pytest.mark.parametrize('s, expected_c', _SEQUENCES_TO_TEST)
@pytest.mark.parametrize(
    'p', [2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12] + [19, 31, 127]
)
def test_poly_eea_zalka_lfsr_classical_gen_sequence_opt(s, expected_c, p):
    if max(s) > GF(p).elements[-1]:
        return
    GFM = GF(p)
    expected_c = GFM([abs(x) for x in expected_c])
    s = generate_sequence(GFM(s[: len(expected_c)]), len(s), expected_c)
    assert_valid_recurrence(s, expected_c)
    c = poly_eea_zalka_lfsr(s)
    assert_valid_recurrence(s, c[1:])


@pytest.mark.parametrize('s, expected_c', _SEQUENCES_TO_TEST)
@pytest.mark.parametrize(
    'p', [2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12] + [19, 31, 127]
)
def test_poly_eea_zalka_lfsr_sync_classical_gen_sequence_opt(s, expected_c, p):
    if max(s) > GF(p).elements[-1]:
        return
    GFM = GF(p)
    expected_c = GFM([abs(x) for x in expected_c])
    s = generate_sequence(GFM(s[: len(expected_c)]), len(s), expected_c)
    assert_valid_recurrence(s, expected_c)
    c = poly_eea_zalka_lfsr_sync(s)
    assert_valid_recurrence(s, c[1:])


import galois


def generate_worst_case_poly(n, field=None):
    """
    Generates two polynomials, A(z) and B(z), using the `galois` library
    that represent a worst-case input for the Extended Euclidean Algorithm.

    The worst case is achieved when the quotient polynomial at each step of
    the EEA is of degree 1, forcing the remainder's degree to decrease
    by exactly 1 in each iteration.

    This function constructs the polynomials by working backward from the end of
    the algorithm, setting the quotient q(z) = z at each step.

    Args:
        n (int): The desired degree of the larger polynomial, A(z). Must be > 0.
        field (galois.FieldClass, optional): The finite field to perform
            arithmetic in. If None, calculations are done over the integers.
            Defaults to None.

    Returns:
        tuple: A tuple containing two galois.Poly objects (A, B)
               where deg(A) = n and deg(B) = n - 1.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Degree 'n' must be a positive integer.")

    # Let the quotient at each step be q(z) = z
    # The list [1, 0] represents 1*z^1 + 0*z^0
    q = galois.Poly([1, 0], field=field)

    # We start with the base polynomials of degree 0 and 1.
    # Initialize with the n=1 case: A(z)=z, B(z)=1
    b_poly = galois.Poly([1], field=field)  # Polynomial of degree n-1
    a_poly = galois.Poly([1, 0], field=field)  # Polynomial of degree n

    # We already have the case for n=1. We loop n-1 times to reach degree n.
    for _ in range(1, n):
        # The backward recurrence: r_{i-2} = q * r_{i-1} + r_i
        # galois.Poly overloads the standard operators for polynomial arithmetic.
        next_a = q * a_poly + b_poly

        # Update for the next iteration
        b_poly = a_poly
        a_poly = next_a

    return a_poly, b_poly


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 7, 20, 50, 100, 200])
def test_worst_case_inputs_gcd(n):
    field = GF(2, 8)
    A, B = generate_worst_case_poly(n, field)
    expected_g = galois.gcd(A, B)
    assert poly_eea_zalka_gcd(A, B) == expected_g
    g = poly_eea_zalka_gcd_sync(A, B)
    assert g == expected_g
