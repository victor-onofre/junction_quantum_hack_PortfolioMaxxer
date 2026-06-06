"""Benchmark for :func:`find_subspace_evasive_set`.

This module provides a small timing benchmark for the heuristic
``find_subspace_evasive_set`` search.  The benchmark uses ``n = 7`` and
``w = 4`` as a representative problem size and prints the elapsed wall time
for a single run.
"""

from __future__ import annotations

import time

from dqi.bitwise import find_subspace_evasive_set


def benchmark(n: int = 7, w: int = 4, restarts: int = 20, seed: int = 0) -> None:
    """Run the benchmark and print the elapsed time.

    Parameters match :func:`find_subspace_evasive_set`.
    """
    start = time.perf_counter()
    result = find_subspace_evasive_set(n, w, restarts, seed)
    elapsed = time.perf_counter() - start
    print(
        f"find_subspace_evasive_set({n}, {w}, {restarts}, {seed}) took {elapsed:.3f}s"
        f" and produced {len(result)} points"
    )
    print(sorted(result))


if __name__ == "__main__":  # pragma: no cover - manual benchmarking helper
    benchmark(n=10, w=7, restarts=1, seed=2342348762)
