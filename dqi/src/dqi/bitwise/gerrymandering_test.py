import pytest
from dqi.bitwise.gerrymandering import gerrymandering_ext


def test_maximize_probability_backtracking():
    result = gerrymandering_ext.maximize_probability(
        m=20, b=2, budget=10, target=3, p_s=[0.1, 0.2, 0.3], solver="backtracking"
    )
    assert isinstance(result, dict)
    assert "phi_c" in result
    assert "choices" in result
    assert isinstance(result["phi_c"], float)
    assert isinstance(result["choices"], list)
    assert abs(result["phi_c"] - 0.601569) < 1e-6


def test_maximize_probability_knapsack():
    result = gerrymandering_ext.maximize_probability(
        m=20, b=2, budget=10, target=3, p_s=[0.1, 0.2, 0.3], solver="knapsack"
    )
    assert isinstance(result, dict)
    assert "phi_c" in result
    assert "choices" in result
    assert isinstance(result["phi_c"], float)
    assert isinstance(result["choices"], list)
    assert abs(result["phi_c"] - 0.601569) < 1e-6
