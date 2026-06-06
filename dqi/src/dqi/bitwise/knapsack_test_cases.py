import dqi.bitwise
import numpy as np
import itertools
from dqi.bitwise.gerrymandering import gerrymandering_ext
import math


def dqival(delta, sigma):
    if delta >= 1 - sigma:
        return 1.0

    return (np.sqrt(delta * (1 - sigma)) + np.sqrt(sigma * (1 - delta))) ** 2


def dqival_opi(m, n, b, r):
    sigma = r / 2**b
    delta = n / 2 / m
    return dqival(delta, sigma)


def get_coin_biases(subset_points):
    assert len(subset_points)
    b = len(subset_points[0])
    coin_biases = []
    for k in range(b + 1):
        max_overlap = dqi.bitwise.find_max_overlap_affine_subspace(subset_points, k)
        coin_biases.append(max_overlap / 2**k)
    return coin_biases[::-1]


if __name__ == '__main__':
    subset_points = [
        (0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1, 1),
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 0, 0, 1, 1, 1),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 1, 0, 1, 1),
        (0, 0, 1, 1, 0, 0),
        (0, 0, 1, 1, 0, 1),
        (0, 1, 0, 0, 0, 0),
        (0, 1, 0, 1, 0, 0),
        (0, 1, 0, 1, 1, 0),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 0, 1, 0, 0),
        (1, 0, 0, 1, 1, 0),
        (1, 0, 1, 0, 0, 0),
        (1, 0, 1, 0, 0, 1),
        (1, 0, 1, 0, 1, 0),
        (1, 0, 1, 0, 1, 1),
        (1, 0, 1, 1, 0, 1),
        (1, 0, 1, 1, 1, 0),
        (1, 1, 0, 0, 0, 1),
        (1, 1, 0, 0, 1, 1),
        (1, 1, 0, 1, 0, 0),
        (1, 1, 0, 1, 1, 0),
        (1, 1, 0, 1, 1, 1),
        (1, 1, 1, 0, 1, 0),
        (1, 1, 1, 0, 1, 1),
        (1, 1, 1, 1, 1, 1),
    ]
    coin_biases = get_coin_biases(subset_points)
    print(coin_biases)
    b = len(coin_biases) - 1
    m = 10
    n = 5
    for solver in ['backtracking', 'knapsack']:
        print(solver)
        result = gerrymandering_ext.maximize_probability(
            m=m, b=b * n, budget=10, target=10, p_s=coin_biases, solver="backtracking"
        )
        print(result)

    # Try a random subset of {0,1}^9 of size 2^7
    b = 9
    r = 200
    F2b = list(itertools.product([0, 1], repeat=b))
    subset_points = [F2b[i] for i in np.random.choice(np.arange(len(F2b)), replace=False, size=r)]
    coin_biases = get_coin_biases(subset_points)
    print(coin_biases)
    m = 10
    n = 5
    t = dqival_opi(m, n, b, r) * m
    print(f'target = {t}')
    for solver in ['backtracking', 'knapsack']:
        print(solver)
        result = gerrymandering_ext.maximize_probability(
            m=m, b=b, budget=b * n, target=math.ceil(t), p_s=coin_biases, solver="backtracking"
        )
        print(result)

    # Row 4 of Tanuj's table:
    m = 125
    n = 20
    t = dqival_opi(m, n, b, r) * m
    print(f'm = {m} n = {n} r = {r} target = {t} knapsack solver:')
    result = gerrymandering_ext.maximize_probability(
        m=m, b=b, budget=b * n, target=math.ceil(t), p_s=coin_biases, solver="knapsack"
    )
    print(result)
