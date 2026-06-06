import numpy as np
import math
import pytest
import galois

from galois import Poly, GF

from dqi.gcd.divstep_gcd import divstep_gcd_int, divstep_gcd_poly_gf, divstep_egcd_poly_gf


def test_gcd_int_brute_force():
    for a in range(1, 4000, 2):
        for b in range(1, a, 1):
            g = math.gcd(a, b)
            assert divstep_gcd_int(a, b) == g


def test_gcd_int_random():
    n_iters = 10_000
    rs = np.random.default_rng(42)
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
            assert divstep_gcd_int(a, b) == g


def test_poly_eea_divstep_small():
    p = 7
    gf = GF(7)
    r0 = Poly(np.array([2, 7, 1, 8, 2, 8, 1, 8]) % p, field=gf)
    r1 = Poly(np.array([3, 1, 4, 1, 5, 9, 2]) % p, field=gf)
    _gcd = divstep_gcd_poly_gf(r1, r0)
    assert _gcd == galois.gcd(r0, r1)
    (x, y, g) = divstep_egcd_poly_gf(r0, r1)
    assert r0 * x + r1 * y == g == _gcd
    r = Poly(np.array([2, 3]), field=gf)
    r0, r1 = r0 * r, r1 * r
    _gcd = divstep_gcd_poly_gf(r1, r0)
    assert _gcd == galois.gcd(r0, r1)
    (x, y, g) = divstep_egcd_poly_gf(r0, r1)
    assert r0 * x + r1 * y == g == _gcd


@pytest.mark.parametrize(
    'p', [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**15, 2**20, 2**30, 2**40] + [7, 19, 31, 127]
)
def test_poly_eea_divstep(p: int):
    rs = np.random.default_rng(p)
    gf = GF(p)
    logp = int(np.log2(p)) + 1
    for _ in range(5):
        roots = gf.Range(1, p) if p <= 100 else gf.Random(100, seed=rs)
        polys = np.array([Poly([1, r], field=gf) for r in roots])
        # Balanced A and B polynomials
        A = np.prod(polys[: len(polys) // 2 + 3], initial=Poly([1], field=gf))
        B = np.prod(polys[len(polys) // 2 :], initial=Poly([1], field=gf))
        assert A.degree > B.degree
        expected_g = galois.gcd(A, B)
        print(f'One: {A=}\n{B=}')
        g = divstep_gcd_poly_gf(Poly(A.coeffs), Poly(B.coeffs))
        assert g == expected_g
        print(f'Two: {A=}\n{B=}')
        (x, y, g) = divstep_egcd_poly_gf(A, B)
        assert x * A + y * B == g == expected_g
        # Skewed A and B polynomials
        A = np.prod(polys[logp - 2 :], initial=Poly([1], field=gf))
        B = np.prod(polys[: logp - 1], initial=Poly([1], field=gf))
        if A.degree < B.degree:
            A, B = B, A
        expected_g = galois.gcd(A, B)
        g = divstep_gcd_poly_gf(A, B)
        assert g == expected_g
        (x, y, g) = divstep_egcd_poly_gf(A, B)
        assert x * A + y * B == g == expected_g
