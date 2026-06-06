#!/usr/bin/env python3

import sys
import numpy as np
import scipy


def dqigreedy(k, D, p):
    assert all((D - 2 * j) >= 0 for j in range(D // 2 + 1))
    return (
        p
        + sum(
            (D - 2 * j) * scipy.special.comb(D, j) * p**j * (1 - p) ** (D - j)
            for j in range(D // 2 + 1)
        )
        * k
        / D
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Usage: dqival.py k D p")
    k = int(sys.argv[1])
    D = int(sys.argv[2])
    p = float(sys.argv[3])
    print(dqigreedy(k, D, p))
