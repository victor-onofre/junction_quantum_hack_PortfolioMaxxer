from galois import Poly


def n_iterations(a, b):
    return 2 * max(a.bit_length(), b.bit_length())


def binary_gcd_int(a: int, b: int) -> int:
    g = 1
    # Preprocessing.
    while a & 1 == 0 and b & 1 == 0:
        g, a, b = g << 1, a >> 1, b >> 1
    while a & 1 == 0:
        a >>= 1
    while b & 1 == 0:
        b >>= 1
    # Both a & b are assumed to be odd.
    for _ in range(n_iterations(a, b)):
        is_a_lt_b = a < b
        is_a_odd = a & 1
        if is_a_odd and is_a_lt_b:
            # gcd(a, b) = gcd((b - a)/2, a)
            a, b = b, a
            a -= b
            a >>= 1
        elif is_a_odd:
            # gcd(a, b) = gcd((a - b)/2, b)
            a -= b
            a >>= 1
        else:
            # gcd(a, b) = gcd(a / 2, b)
            a >>= 1
    return b * g


def binary_gcd_poly_gf(a: Poly, b: Poly) -> Poly:
    gf = a.field
    g, x, zero = Poly.One(gf), Poly.Identity(gf), Poly.Zero(gf)
    # Preprocessing.
    while a(0) == 0 and b(0) == 0:
        g, a, b = g * x, a // x, b // x
    while a(0) == 0:
        a //= x
    while b(0) == 0:
        b //= x
    # Algo starts. Both a & b are assumed to have non-zero const term.
    n_iterations = 2 * max(a.degree, b.degree)
    for _ in range(n_iterations):
        if a(0) != 0:
            if a.degree < b.degree:
                a, b = b, a
            a -= a(0) * (b(0) ** -1) * b
        a //= x
    return (b * g) // b.coeffs[0]
