import time
import math
import numpy as np
from dqi.bitwise import find_subspace_evasive_set, affine_spectrum
import dqi.bitwise
from knapsack_test_cases import get_coin_biases, dqival_opi
from dqi.bitwise.gerrymandering import gerrymandering_ext
import itertools
import argparse

parser = argparse.ArgumentParser('do a resource estimate for some m, n, b')

parser.add_argument('--b', type=int, required=True)
parser.add_argument('--m', type=int, required=True)
parser.add_argument('--n', type=int, required=True)
args = parser.parse_args()

b = args.b
assert b % 2 == 0, 'b must be even'


def boolfun(bs):
    total = 0
    h = len(bs) // 2
    for i in range(h):
        total += bs[i] * bs[i + h]
    return total % 2


result = [bs for bs in itertools.product([0, 1], repeat=b) if boolfun(bs) == 1]
print(f'got {len(result)} points')
# start_time = time.time()
# spectrum = affine_spectrum(result)
# end_time = time.time()
# print(f'affine spectrum = {spectrum} duration {end_time-start_time}')
# print(str(sorted(result)).replace(' ', ''))
# start_time = time.time()
# coin_biases = get_coin_biases(result)
# end_time = time.time()
# print(f'coin_biases = {coin_biases} duration {end_time-start_time}')
# I have checked that this ansatz holds at least for b = 8 or 10
ansatz_max_overlaps = []
for k in range(b + 1):
    if k < b / 2:
        # For small subspaces, a 100% overlap is possible
        val = 2**k
    elif k == b:
        # The entire space; intersection is the support size
        val = 2 ** (b - 1) - 2 ** (b // 2 - 1)
    elif k == b - 1:
        # Hyperplanes are always balanced
        val = 2 ** (b - 2)
    else:  # b/2 <= k < b-1
        # The formula revealed by your data
        val = 2 ** (k - 1) + 2 ** (b // 2 - 2)
    ansatz_max_overlaps.append(val)
ansatz_coin_biases = [overlap / 2**w for w, overlap in enumerate(ansatz_max_overlaps)][::-1]
print(f'ansatz_coin_biases = {ansatz_coin_biases}')
coin_biases = ansatz_coin_biases

m = args.m
# for n in range(3, min(m-1, 151), 1):
# for n in range(70, min(m-1, 151), 1):
n = args.n
t = dqival_opi(m, n, b, len(result)) * m
target = round(t)
# target = math.ceil(t)
target = min(target, m)
allocation = gerrymandering_ext.maximize_probability(
    m=m, b=b, budget=b * n, target=target, p_s=coin_biases, solver="knapsack", fast=True
)
print(
    f'n = {n} m = {m} target = {target} phi_c = {allocation["phi_c"]} estimated classical trials = {1/allocation["phi_c"]}'
)
