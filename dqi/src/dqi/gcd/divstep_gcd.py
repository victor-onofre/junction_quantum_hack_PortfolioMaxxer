import numpy as np
from galois import Poly


def n_iterations(flen: int, glen: int):
    # Compute number of iterations per :
    # https://eprint.iacr.org/2021/549.pdf page 14
    # https://github.com/sipa/safegcd-bounds
    d = max(flen, glen)
    return int(np.ceil((45907 * d + 26313) / 19929)) + 2


def divstep_gcd_int(f: int, g: int) -> int:
    delta = 1
    assert f & 1
    for i in range(2 * n_iterations(f.bit_length(), g.bit_count())):
        is_swap = delta > 0 and g & 1
        if is_swap:
            # Note: Negating `f` here leads to faster convergence and
            # thus needs n_iterations instead of 2x that. Not sure why.
            delta, f, g = -delta, g, f
            assert g & 1
            g = g - f
        elif g & 1:
            g = g - f
        assert g & 1 == 0
        delta = 1 + delta
        g = g >> 1
    return abs(f)


def divstep_gcd_poly_gf(a: Poly, b: Poly) -> Poly:
    gf = a.field
    g, x, zero = Poly.One(gf), Poly.Identity(gf), Poly.Zero(gf)
    while a(0) == 0 and b(0) == 0:
        g, a, b = g * x, a // x, b // x
    # Algo starts. One of a or b must have non-zero const term.
    delta = 1
    for i in range(n_iterations(a.degree, b.degree)):
        is_swap = delta > 0 and b(0) != 0
        if is_swap:
            delta, a, b = -delta, b, a
        delta = 1 + delta
        b = b - a * (b(0) // a(0))
        assert b(0) == 0
        b //= x
    return (a * g) // a.coeffs[0]


def divstep_egcd_poly_gf(a: Poly, b: Poly) -> tuple[Poly, Poly, Poly]:
    gf = a.field
    assert a.degree > b.degree
    x, zero, d_a, d_b = Poly.Identity(gf), Poly.Zero(gf), a.degree, b.degree
    # Instead of maintaining Bezout coefficient polynomials as u_0 + u_1/x + u_2/x^2 + ...
    # we take x^{d} common and store u_0x^d + u_1x^{d-1} + ...; this requires us to also
    # reverse the `a` and `b` polynomials.
    # In the end, we'll have a * x_i + b * y_i = _gcd. x_i and x_j maintain the current and
    # previous values for coefficient `x_i` and y_i and y_j maintain the current and previous
    # values for coefficient `y_i`.
    n_a, n_b, delta = d_a, d_b, d_a - d_b
    a, b = a.reverse(), b.reverse()
    x_i, x_j, y_i, y_j = Poly.One(gf), Poly.Zero(gf), Poly.Zero(gf), Poly.One(gf)
    for _ in range(2 * d_a + 1):
        is_swap = delta > 0 and b(0) != 0
        if is_swap:
            delta, a, b = -delta, b, a
            x_i, x_j, y_i, y_j = x_j, x_i, y_j, y_i
            n_a, n_b = n_b, n_a
        delta = 1 + delta
        coeff = b(0) // a(0)
        assert n_a <= n_b or coeff == 0, f'{n_a=}, {n_b=}'
        b = (b - a * coeff) // x
        x_j = (x_j - x_i * coeff) * x
        y_j = (y_j - y_i * coeff) * x
        n_b -= 1
    # Degree of the gcd is `n_a = (delta - delta_orig) // 2` at this point.
    assert n_a == (delta - (d_a - d_b)) // 2
    assert x_i % x ** (d_a - n_a) == 0 and y_i % x ** (d_b - n_a) == 0
    x_i, y_i = x_i // x ** (d_a - n_a), y_i // x ** (d_b - n_a)
    _gcd = a.reverse() * x ** (n_a - a.degree)
    return x_i // a(0), y_i // a(0), _gcd // a(0)
