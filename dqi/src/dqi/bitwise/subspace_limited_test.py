import pytest
from dqi.bitwise import find_subspace_limited_set, affine_spectrum
import itertools


def get_affine_subspaces(n, w):
    """Helper to generate all affine subspaces for testing."""
    if w < 0 or w > n:
        return

    # Generate all linear subspaces (basis in RREF)
    for pivots_tuple in itertools.combinations(range(n), w):
        pivots = list(pivots_tuple)
        free_cols = [i for i in range(n) if i not in pivots]

        num_free_vars_in_rref = w * (n - w) - sum(pivots)

        # Iterate through all possible RREF matrices for these pivots
        for rref_fill in range(1 << num_free_vars_in_rref):
            temp_fill = rref_fill
            basis = []
            for i in range(w):
                row = 1 << pivots[i]
                for j in range(i + 1, w):
                    if temp_fill & 1:
                        row |= 1 << pivots[j]
                    temp_fill >>= 1
                for j_idx, j in enumerate(free_cols):
                    if temp_fill & 1:
                        row |= 1 << j
                    temp_fill >>= 1
                basis.append(row)

            # Generate the linear subspace
            linear_subspace = set()
            for i in range(1 << w):
                vec = 0
                for j in range(w):
                    if (i >> j) & 1:
                        vec ^= basis[j]
                linear_subspace.add(vec)

            # Generate all affine shifts
            for shift in range(1 << (n - w)):
                affine_subspace = set()
                shift_vec = 0
                for i, free_col in enumerate(free_cols):
                    if (shift >> i) & 1:
                        shift_vec |= 1 << free_col

                for vec in linear_subspace:
                    affine_subspace.add(vec ^ shift_vec)
                yield affine_subspace


@pytest.mark.parametrize("n, max_overlaps", [(3, [1, 2, 3, 4]), (4, [1, 1, 2, 3, 9])])
def test_find_subspace_limited_set(n: int, max_overlaps: list[int]) -> None:
    result_tuples = find_subspace_limited_set(n, max_overlaps, 10, 0)
    result_set = set()
    for t in result_tuples:
        val = 0
        for i, bit in enumerate(t):
            if bit:
                val |= 1 << i
        result_set.add(val)

    # Verification
    for w in range(n + 1):
        if max_overlaps[w] > (1 << w):
            continue  # Vacuously true

        spectrum = affine_spectrum(result_tuples)
        for i in range(max_overlaps[w] + 1, len(spectrum[w])):
            assert spectrum[w][i] == 0, f"Overlap constraint failed for w={w}"
