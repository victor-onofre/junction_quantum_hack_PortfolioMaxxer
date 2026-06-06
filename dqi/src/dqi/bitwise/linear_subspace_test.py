from dqi.bitwise import count_linear_subspaces


def gaussian_binomial(n: int, k: int) -> int:
    num = 1
    den = 1
    for i in range(k):
        num *= (1 << (n - i)) - 1
        den *= (1 << (k - i)) - 1
    return num // den


def test_linear_subspaces_small_cases():
    for n in range(1, 5):
        for k in range(n + 1):
            expected = gaussian_binomial(n, k)
            assert count_linear_subspaces(k, n) == expected
