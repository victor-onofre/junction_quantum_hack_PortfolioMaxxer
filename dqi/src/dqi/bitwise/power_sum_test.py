from dqi.bitwise import power_sum


def test_power_sum_square():
    assert power_sum(4, 2) == 30.0


def test_power_sum_linear():
    assert power_sum(5, 1) == 15.0
