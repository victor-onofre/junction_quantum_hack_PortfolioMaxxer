#!/usr/bin/env python3

import sys
import numpy as np


def dqival(delta, sigma):
    if delta >= 1 - sigma:
        return 1.0

    return (np.sqrt(delta * (1 - sigma)) + np.sqrt(sigma * (1 - delta))) ** 2


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise ValueError("Usage: dqival.py DELTA (sigma)")
    delta = float(sys.argv[1])
    sigma = 0.5
    if len(sys.argv) >= 3:
        sigma = float(sys.argv[2])

    print(dqival(delta, sigma))
