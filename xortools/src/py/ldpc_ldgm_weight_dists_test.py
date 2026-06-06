import pytest
import numpy as np
import galois
import scipy
from ldpc_ldgm_weight_dists import macwilliams_transform_krawtchouk
import itertools


def iterate_over_span(B):
    for bs in itertools.product([0, 1], repeat=B.shape[0]):
        yield np.dot(B.T, galois.GF2(bs))


def weight(y):
    return np.sum(y == 1)


def weight_dist(B):
    dist = np.zeros(B.shape[1] + 1, dtype=int)
    for y in iterate_over_span(B):
        dist[weight(y)] += 1
    return dist


@pytest.mark.parametrize("n, m", [(3, 20), (17, 20), (10, 20), (8, 20)])
def test_macwilliams_transform_krawtchouk(n, m):
    """
    Test macwilliams_transform_krawtchouk function for correctness using random binary matrices.

    Args:
        m (int): Number of rows in the parity-check matrix.
        n (int): Number of columns in the parity-check matrix.
    """
    # Generate a random binary parity-check matrix H of shape m x n
    np.random.seed(10345834)
    H = np.random.randint(0, 2, size=(n, m), dtype=np.uint8)
    H = galois.GF2(H)
    rank = np.linalg.matrix_rank(H)
    assert rank == n
    kernel = H.null_space()
    span_weight_dist = weight_dist(H)
    ker_weight_dist = weight_dist(kernel)
    print(span_weight_dist)
    print(ker_weight_dist)
    assert (
        macwilliams_transform_krawtchouk(m, m - n, span_weight_dist) == ker_weight_dist
    ).all()
    assert (
        macwilliams_transform_krawtchouk(m, n, ker_weight_dist) == span_weight_dist
    ).all()
