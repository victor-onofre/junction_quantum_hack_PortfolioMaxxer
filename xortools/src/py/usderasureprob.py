#!/usr/bin/env python3

import sys
import numpy as np


def perp(x, p):
  return (np.sqrt((1-x)*(p-1)) - np.sqrt(x))**2 / p

def erasure_rate_old(delta, p):
  return 1 - p * perp(delta, p) / (p-1)

def erasure_rate(delta, sigma):
  return ((np.sqrt(delta * (1 - sigma)) + np.sqrt(sigma * (1 - delta))) ** 2 - sigma)/(1-sigma)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: dqival.py DELTA (q) (r)")
    delta = float(sys.argv[1])
    q = 2
    if len(sys.argv) > 2:
        q = int(sys.argv[2])
    r = 1
    if len(sys.argv) > 3:
        r = int(sys.argv[3])
    sigma = r/q
    print(erasure_rate(delta, sigma))
