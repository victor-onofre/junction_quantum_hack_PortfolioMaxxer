"""Sample script demonstrating the C extension."""

import os
import sys

# Allow running as `python dqi/bitwise/main.py` from repo root.
if __package__ is None:  # pragma: no cover - runtime convenience
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import dqi.bitwise
from dqi.bitwise import (
    power_sum,
    count_linear_subspaces,
    find_hitting_subspace,
    find_subspace_evasive_set,
)


def demo() -> None:
    N, i = 5, 2
    total = power_sum(N, i)
    print(f"sum_{{n=1..{N}}} n^{i} = {total}")

    # Example: number of 1-dim subspaces of F_2^3
    lines = count_linear_subspaces(1, 3)
    print(f"number of lines through 0 in F_2^3: {lines}")

    from contains_affine_subspace_test import (
        claimed_subset_of_F_2_8_avoiding_all_4d_affine_subspaces as claimed,
    )

    # Find a 1D hitting subspace of the claimed set.
    result = find_hitting_subspace(claimed, 1)
    print(result)

    # Search for a larger 4-subspace-evasive subset of F_2^8.
    """Sample script demonstrating the C extension."""


import os
import sys

# Allow running as `python dqi/bitwise/main.py` from repo root.
if __package__ is None:  # pragma: no cover - runtime convenience
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import dqi.bitwise
from dqi.bitwise import (
    power_sum,
    count_linear_subspaces,
    find_hitting_subspace,
    find_subspace_evasive_set,
)


def demo() -> None:
    N, i = 5, 2
    total = power_sum(N, i)
    print(f"sum_{{n=1..{N}}} n^{i} = {total}")

    # Example: number of 1-dim subspaces of F_2^3
    lines = count_linear_subspaces(1, 3)
    print(f"number of lines through 0 in F_2^3: {lines}")

    from contains_affine_subspace_test import (
        claimed_subset_of_F_2_8_avoiding_all_4d_affine_subspaces as claimed,
    )

    # Find a 1D hitting subspace of the claimed set.
    result = find_hitting_subspace(claimed, 1)
    print(result)

    # Search for a larger 4-subspace-evasive subset of F_2^8.
    bigger = find_subspace_evasive_set(8, 4, max_restarts=200, seed=1)
    print(f"heuristic search found set of size {len(bigger)} versus {len(claimed)}")


if __name__ == "__main__":
    demo()

    print(f"heuristic search found set of size {len(bigger)} versus {len(claimed)}")


if __name__ == "__main__":
    demo()
