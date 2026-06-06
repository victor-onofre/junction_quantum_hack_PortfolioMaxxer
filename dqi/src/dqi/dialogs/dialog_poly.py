from typing import Optional
from galois import Poly, GF, gcd


def n_iterations(d):
    return 2 * d


def construct_dialog_poly_divstep(f: Poly, g: Poly, n_iters=None) -> tuple[Poly, list[GF]]:
    delta = f.degree - g.degree
    d = 1 + max(f.degree, g.degree)
    x = Poly.Identity(f.field)
    dialog = []
    assert f(0) != 0
    if n_iters is None:
        n_iters = n_iterations(d)
    for _ in range(n_iters):
        is_swap = delta > 0 and g(0) != 0
        if is_swap:
            delta, f, g = -delta, g, f
        dialog.append(g(0) // f(0))
        g = g - f * g(0) // f(0)
        delta = 1 + delta
        assert g(0) == 0
        g = g // x
    assert g == 0
    return f, dialog


def _get_delta(delta: int, text: list[GF]):
    for c in text:
        is_swap = delta > 0 and c != 0
        if is_swap:
            delta = -delta
        delta = 1 + delta
    return delta


def multiply_poly_with_divstep_dialog(
    a: Poly, dialog: list[GF], delta_init: int, mod: Optional[Poly] = None
) -> tuple[Poly, Poly]:
    f, g, x = a, Poly.Zero(a.field), Poly.Identity(a.field)
    for i, c in [*enumerate(dialog)][::-1]:
        is_swap = _get_delta(delta_init, dialog[:i]) > 0 and c != 0
        g = (g * x) % mod if mod else g * x
        if c != 0:
            g = (g + c * f) % mod if mod else g + c * f
        if is_swap:
            f, g = g, f
    return f, g


def mod_divide_poly_with_divstep_dialog(
    a: Poly, dialog: list[GF], delta_init: int, mod: Poly
) -> tuple[Poly, Poly]:
    f, g, x = Poly.Zero(a.field), a, Poly.Identity(a.field)
    gf = GF(a.field.characteristic, len(dialog) // 2 - 1, irreducible_poly=mod)
    x_inv = (x ** (gf.order - 2)) % mod
    assert (x * x_inv) % mod == Poly.One(a.field)
    for i, c in enumerate(dialog):
        is_swap = _get_delta(delta_init, dialog[:i]) > 0 and c != 0
        if is_swap:
            f, g = g, f
        if c != 0:
            g = (g - c * f) % mod
        g = (g * x_inv) % mod
    return f, g


def construct_dialog_poly_divstep_inplace(f: Poly, g: Poly) -> tuple[Poly, list[GF]]:
    gf, n, x = f.field, 1 + max(f.degree, g.degree), Poly.Identity(f.field)
    # In general, we need upto 3n space to store upto n coefficients of the gcd.
    # If we know that gcd(f, g) = 1, the length of poly will reduce to 2n + 3.
    _gcd_degree = gcd(f, g).degree
    poly = [gf(0) for _ in range(2 * n + 3 + _gcd_degree)]
    poly[: f.degree + 1] = f.coefficients(order="asc")
    poly[-g.degree - 1 :] = g.coefficients(order="desc")
    delta = f.degree - g.degree
    dialog = []
    assert poly[0] != 0
    for _ in range(2 * n):
        is_swap = delta > 0 and poly[-1] != 0
        if is_swap:
            delta, poly = -delta, poly[::-1]
        coeff = poly[-1] // poly[0]
        dialog.append(coeff)
        delta = 1 + delta
        # Perform g = (g - f * coeff) // x
        for j in range(len(poly) // 2 - 1, -1, -1):
            if j < len(poly) // 2 + delta // 2:
                poly[-j - 1] -= poly[j] * coeff
        assert poly.pop() == 0
    return Poly(poly[: 1 + (delta - 1) // 2][::-1], field=f.field), dialog
