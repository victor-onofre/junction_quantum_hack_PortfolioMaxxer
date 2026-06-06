import pytest
from dqi.bitwise import affine_spectrum


def test_affine_spectrum_simple():
    # For n=2, F_2^2 = {0, 1, 2, 3}.
    # Let S = {0, 1}.
    # The C extension expects a list of lists of bits.
    # Point 0 is (0,0), Point 1 is (1,0)
    S = [[0, 0], [1, 0]]

    spectrum = affine_spectrum(S)

    # n=2, so we expect spectrum for k=0, 1, 2.
    assert len(spectrum) == 3

    # k=0: 0-dim subspaces are points. There are 4 points.
    # Subspaces: {0}, {1}, {2}, {3}
    # Intersection with S:
    # {0}: size 1
    # {1}: size 1
    # {2}: size 0
    # {3}: size 0
    # Spectrum[0]: s=0: 2 times, s=1: 2 times.
    # Expected: [count_s0, count_s1] = [2, 2]
    assert len(spectrum[0]) == 2**0 + 1
    assert spectrum[0] == [2, 2]

    # k=1: 1-dim subspaces are lines with 2 points.
    # There are 6 such affine subspaces.
    # Linear subspaces:
    # L1 = {0,1} = {(0,0), (1,0)}. Cosets: {0,1}, {2,3}={(0,1),(1,1)}
    #      Intersections with S: {0,1} -> 2, {2,3} -> 0
    # L2 = {0,2} = {(0,0), (0,1)}. Cosets: {0,2}, {1,3}
    #      Intersections with S: {0,2} -> 1, {1,3} -> 1
    # L3 = {0,3} = {(0,0), (1,1)}. Cosets: {0,3}, {1,2}
    #      Intersections with S: {0,3} -> 1, {1,2} -> 1
    # Spectrum[1]: s=0: 1, s=1: 4, s=2: 1
    # Expected: [count_s0, count_s1, count_s2] = [1, 4, 1]
    assert len(spectrum[1]) == 2**1 + 1
    assert spectrum[1] == [1, 4, 1]

    # k=2: 2-dim subspace is the whole space.
    # There is 1 such subspace: {0, 1, 2, 3}
    # Intersection with S is 2.
    # Spectrum[2]: s=2: 1
    # Expected: [c_s0, c_s1, c_s2, c_s3, c_s4] = [0, 0, 1, 0, 0]
    assert len(spectrum[2]) == 2**2 + 1
    assert spectrum[2] == [0, 0, 1, 0, 0]


def test_affine_spectrum_properties():
    # For any S, the sum of counts for a given k must be the total
    # number of k-dimensional affine subspaces.
    n = 3
    S = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]

    spectrum = affine_spectrum(S)

    # Total number of affine subspaces of dimension k in F_2^n is
    # 2^(n-k) * [n choose k]_2, where [n choose k]_2 is the q-binomial
    # coefficient with q=2.
    # [n,k]_2 = product_{i=0}^{k-1} (2^n - 2^i) / (2^k - 2^i)

    # For n=3:
    # k=0: 2^3 * [3,0]_2 = 8 * 1 = 8. Subspaces are points.
    # k=1: 2^2 * [3,1]_2 = 4 * (2^3-1)/(2^1-1) = 4 * 7 = 28.
    # k=2: 2^1 * [3,2]_2 = 2 * (2^3-1)(2^3-2)/(2^2-1)(2^2-2) = 2 * (7*6)/(3*2) = 14.
    # k=3: 2^0 * [3,3]_2 = 1 * 1 = 1.

    num_k_subspaces = [8, 28, 14, 1]

    for k in range(n + 1):
        assert sum(spectrum[k]) == num_k_subspaces[k]


def test_empty_set():
    # The C code currently fails on empty set as it cannot determine `n`.
    # This is a known issue with the current implementation.
    # We expect a ValueError.
    with pytest.raises(ValueError):
        affine_spectrum([])
