import pytest
from galois import GF

from dqi.dialogs.dialog_int import (
    construct_binary,
    parse_binary,
    construct_dialog_int_divstep,
    multiply_int_with_divstep_dialog,
    mod_divide_int_with_divstep_dialog,
    construct_dialog_int_bin,
    multiply_int_with_bin_dialog,
    mod_divide_int_with_bin_dialog,
)


def test_construct_and_parse_binary():
    for x in range(1000):
        x_bin = construct_binary(x)
        assert x_bin == bin(x)[2:][::-1][: x.bit_length()]
        assert parse_binary(x_bin) == x


def test_construct_and_parse_dialog_divstep():
    for x in range(1, 1000, 2):
        for y in range(1, 1000):
            _gcd, dialog = construct_dialog_int_divstep(x, y)
            assert multiply_int_with_divstep_dialog(abs(_gcd), dialog) == (x, y)


@pytest.mark.parametrize('p', [3, 5, 7, 11, 19, 31, 127])
def test_multiply_and_divide_using_divstep_dialog(p: int):
    gf = GF(p)
    for x in gf.elements[1:]:
        _gcd, dialog = construct_dialog_int_divstep(p, int(x))
        for y in gf.elements[1:]:
            # Multiply
            x_out, y_out = multiply_int_with_divstep_dialog(y, dialog)
            assert x_out == 0
            assert y_out == x * y or y_out == -x * y
            # Divide
            x_out, y_out = mod_divide_int_with_divstep_dialog(y, dialog)
            assert y_out == 0
            assert x_out == y // x or x_out == -y // x


def test_construct_and_parse_dialog_bin():
    for x in range(1, 1000, 2):
        for y in range(1, 1000):
            _gcd, dialog = construct_dialog_int_bin(x, y)
            assert multiply_int_with_bin_dialog(_gcd, dialog) == (x, y)


@pytest.mark.parametrize('p', [3, 5, 7, 11, 19, 31, 127])
def test_multiply_and_divide_using_bin_dialog(p: int):
    gf = GF(p)
    for x in gf.elements[1:]:
        _gcd, dialog = construct_dialog_int_bin(p, int(x))
        for y in gf.elements[1:]:
            # Multiply
            x_out, y_out = multiply_int_with_bin_dialog(y, dialog)
            assert x_out == 0
            assert y_out == x * y
            # Divide
            x_out, y_out = mod_divide_int_with_bin_dialog(y, dialog)
            print(f'{x_out=}, {y_out=}, {x=}, {y=}, {y // x=}, {_gcd=}')
            assert y_out == 0
            assert x_out == y // x
