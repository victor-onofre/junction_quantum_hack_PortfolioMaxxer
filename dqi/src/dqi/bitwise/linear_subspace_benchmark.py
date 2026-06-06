"""Timing benchmark for the count_linear_subspaces function."""

from __future__ import annotations

from timeit import timeit

from dqi.bitwise import count_linear_subspaces


def benchmark(k: int = 4, n: int = 8, repeats: int = 1000) -> None:
    """Run a simple timing benchmark for count_linear_subspaces.

    Args:
        k: Dimension of the subspace.
        n: Ambient dimension.
        repeats: Number of times to call the function.
    """
    duration = timeit(lambda: count_linear_subspaces(k, n), number=repeats)
    per_call = duration / repeats
    print(
        f"{repeats} runs of count_linear_subspaces({k}, {n}) took {duration:.6f} seconds"
        f" ({per_call:.6e} s per call)"
    )


if __name__ == "__main__":
    benchmark(n=22, k=11, repeats=10)
