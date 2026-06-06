"""C-backed utilities for the dqi package."""

from .c_discrete_math import (
    power_sum,
    count_linear_subspaces,
    count_affine_subspaces,
    contains_affine_subspace,
    find_max_overlap_affine_subspace,
    find_all_max_overlap_affine_subspaces,
    find_hitting_subspace,
    find_subspace_evasive_set,
    find_subspace_limited_set,
    affine_spectrum,
)

__all__ = [
    "power_sum",
    "count_linear_subspaces",
    "count_affine_subspaces",
    "contains_affine_subspace",
    "find_max_overlap_affine_subspace",
    "find_all_max_overlap_affine_subspaces",
    "find_hitting_subspace",
    "find_subspace_evasive_set",
    "find_subspace_limited_set",
    "affine_spectrum",
]
