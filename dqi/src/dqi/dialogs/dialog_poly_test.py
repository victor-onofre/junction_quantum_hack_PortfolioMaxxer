import galois
import pytest
import numpy as np
from galois import GF, Poly

from dqi.dialogs.dialog_poly import (
    construct_dialog_poly_divstep,
    multiply_poly_with_divstep_dialog,
    mod_divide_poly_with_divstep_dialog,
    construct_dialog_poly_divstep_inplace,
)


@pytest.mark.parametrize('m', [3, 5, 8, 12])
@pytest.mark.parametrize('d', [3, 20, 50, 100])
@pytest.mark.parametrize('seed', [*range(1, 50)])
def test_construct_and_parse_poly_divstep(m: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    gf = GF(2, m)
    f = Poly(gf.Random(shape=(d + 1,), seed=rng), field=gf)
    g = Poly(gf.Random(shape=(d + 1,), seed=rng), field=gf)
    if f(0) == 0:
        f += Poly.One(field=gf)
    # Test for (f, g)
    _gcd, dialog = construct_dialog_poly_divstep(f, g)
    delta_init = f.degree - g.degree
    _gcd_inplace, dialog_inplace = construct_dialog_poly_divstep_inplace(f, g)
    assert _gcd == _gcd_inplace, f'{_gcd=}, {_gcd_inplace=}'
    np.testing.assert_array_equal(dialog, dialog_inplace)
    assert _gcd // _gcd.coeffs[0] == galois.gcd(f, g)
    assert multiply_poly_with_divstep_dialog(_gcd, dialog, delta_init) == (f, g)
    # Test for (f', g'), both multiplied by a random polynomial
    factor = Poly(gf.Random(shape=(d + 1,), seed=rng), field=gf)
    if factor(0) == 0:
        factor += Poly.One(field=gf)
    f, g = f * factor, g * factor
    delta_init = f.degree - g.degree
    _gcd, dialog = construct_dialog_poly_divstep(f, g)
    _gcd_inplace, dialog_inplace = construct_dialog_poly_divstep_inplace(f, g)
    assert _gcd == _gcd_inplace, f'{_gcd=}, {_gcd_inplace=}'
    np.testing.assert_array_equal(dialog, dialog_inplace)
    assert _gcd // _gcd.coeffs[0] == galois.gcd(f, g)
    assert multiply_poly_with_divstep_dialog(_gcd, dialog, delta_init) == (f, g)


@pytest.mark.parametrize('gf', [GF(2, 3), GF(2, 8), GF(2, 12), GF(3, 6), GF(7, 5)])
def test_mul_and_div_using_poly_divstep_dialog(gf: GF):
    irred_poly = gf.irreducible_poly
    rs = np.random.default_rng(42)
    x_indices = rs.integers(size=(50,), low=1, high=gf.order - 1)
    y_indices = rs.integers(size=(50,), low=1, high=gf.order - 1)
    all_elements = [*gf.elements[1:]]
    for x_ind in x_indices:
        x = all_elements[x_ind]
        x_poly = Poly(x.vector(), field=GF(gf.characteristic))
        _gcd, x_dialog = construct_dialog_poly_divstep(irred_poly, x_poly)
        _gcd_inplace, x_dialog_inplace = construct_dialog_poly_divstep_inplace(irred_poly, x_poly)
        assert _gcd == _gcd_inplace, f'{_gcd=}, {_gcd_inplace=}'
        np.testing.assert_array_equal(x_dialog, x_dialog_inplace)

        _gcd = Poly([_gcd], field=gf)
        delta_degree = irred_poly.degree - x_poly.degree
        for y_ind in y_indices:
            y = all_elements[y_ind]
            y_poly = Poly(y.vector(), field=GF(gf.characteristic))
            f_poly, g_poly = multiply_poly_with_divstep_dialog(
                y_poly, x_dialog, delta_degree, mod=irred_poly
            )
            assert f_poly == 0 and gf.Vector(g_poly.coefficients(size=gf.degree)) == (x * y) // _gcd

            f_poly, g_poly = mod_divide_poly_with_divstep_dialog(
                y_poly, x_dialog, delta_degree, mod=irred_poly
            )
            assert g_poly == 0 and gf.Vector(f_poly.coefficients(size=gf.degree)) == (y * _gcd) // x
