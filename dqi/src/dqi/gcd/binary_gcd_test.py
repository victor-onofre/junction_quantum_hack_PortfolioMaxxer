import numpy as np
import math
import pytest
import galois

from galois import Poly, GF

from dqi.gcd.binary_gcd import binary_gcd_int, binary_gcd_poly_gf

rs = np.random.default_rng(42)


def test_gcd_int_brute_force():
    for a in range(3, 500):
        for b in range(1, a):
            g = math.gcd(a, b)
            assert binary_gcd_int(a, b) == g


def test_gcd_int_random():
    n_iters = 10_000
    r = rs.integers(low=2**30, high=2**32, size=(n_iters, 2))
    for i, (a, b) in enumerate(r):
        if a < b:
            a, b = b, a
        while a & 1 == 0:
            a //= 2
            b //= 2
        a, b = int(a), int(b)
        for rodd in [1, 3, 2**10 - 1, 2**20 - 1, 2**40 - 1]:
            a, b = rodd * a, rodd * b
            g = math.gcd(a, b)
            assert binary_gcd_int(a, b) == g


def test_poly_eea_binary_small():
    p = 7
    gf = GF(7)
    r0 = Poly(np.array([2, 7, 1, 8, 2, 8, 1, 8]) % p, field=gf)
    r1 = Poly(np.array([3, 1, 4, 1, 5, 9, 2]) % p, field=gf)
    g = binary_gcd_poly_gf(r1, r0)
    assert g == galois.gcd(r0, r1)
    r = Poly(np.array([2, 3]), field=gf)
    r0, r1 = r0 * r, r1 * r
    g = binary_gcd_poly_gf(r1, r0)
    assert g == galois.gcd(r0, r1)


@pytest.mark.parametrize(
    'p', [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**15, 2**20, 2**30, 2**40] + [7, 19, 31, 127]
)
def test_poly_eea_binary(p: int):
    gf = GF(p)
    for _ in range(5):
        logp = int(np.log2(p))
        roots = gf.Random(logp, seed=rs)
        polys = np.array([Poly([1, r], field=gf) for r in roots])
        # Balanced A and B polynomials
        A = np.prod(polys[: logp // 2 + 2], initial=Poly([1], field=gf))
        B = np.prod(polys[logp // 2 :], initial=Poly([1], field=gf))
        expected_g = galois.gcd(A, B)
        g = binary_gcd_poly_gf(A, B)
        assert g == expected_g
        # Skewed A and B polynomials
        A = np.prod(polys[: logp - 1], initial=Poly([1], field=gf))
        B = np.prod(polys[logp - 2 :], initial=Poly([1], field=gf))
        expected_g = galois.gcd(A, B)
        g = binary_gcd_poly_gf(A, B)
        assert g == expected_g
