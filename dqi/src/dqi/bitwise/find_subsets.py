import dqi.bitwise
import multiprocessing as mp
import itertools
import numpy as np
from dqi.bitwise.gerrymandering import gerrymandering_ext
from knapsack_test_cases import get_coin_biases, dqival_opi
from functools import partial

seed = 83459372345
restarts = 10000
num_cores = 64
b = 10


def combined_worker(max_overlaps, b_val, restarts_val, seed_val, m_val, n_range_val):
    """
    A single worker that performs both stages of the computation:
    1. Finds the subspace for a given 'max_overlaps'.
    2. Calculates the allocation probabilities for the entire 'n_range'.
    """
    # --- Stage 1: Find subspace ---
    result = dqi.bitwise.find_subspace_limited_set(b_val, max_overlaps, restarts_val, seed_val)
    coin_biases = get_coin_biases(result)

    # --- Stage 2: Calculate allocations for all n values ---
    inner_results = []
    for n in n_range_val:
        t = round(min(dqival_opi(m_val, n, b_val, len(result)) * m_val, m_val))
        phi_c = gerrymandering_ext.maximize_probability(
            m=m_val,
            b=b_val,
            budget=b_val * n,
            target=t,
            p_s=coin_biases,
            solver="knapsack",
            fast=True,
        )['phi_c']
        inner_results.append({'n': n, 'phi_c': phi_c, 't': t})

    return max_overlaps, len(result), inner_results


# --- Main Script ---


def efficient_options_generator(b, max_deviations=3):
    """
    Efficiently generates lists where up to `max_deviations` elements l[i]
    differ from the base value of 2**i.

    Args:
        b (int): The upper bound for the index 'i'.
        max_deviations (int): The maximum number of elements that can differ
                              from the base list.
    """
    # The base list where all elements l[i] == 2**i
    base_list = [2**i for i in range(b + 1)]

    # Pre-calculate all possible "non-2**i" values for each index.
    # This avoids recalculating them in the inner loops.
    all_defect_vals = [[val for val in range(i + 1, 2**i + 1) if val != 2**i] for i in range(b + 1)]

    # Iterate from 0 deviations up to the maximum allowed.
    for d in range(max_deviations + 1):
        # Case d=0 is just the base list itself.
        if d == 0:
            yield list(base_list)
            continue

        # Choose 'd' unique indices to place the deviations.
        for indices_to_deviate in itertools.combinations(range(b + 1), d):

            # Get the lists of possible deviation values for the chosen indices.
            deviation_options = [all_defect_vals[i] for i in indices_to_deviate]

            # The Cartesian product gives every combination of deviation values
            # for the chosen set of indices.
            for new_values in itertools.product(*deviation_options):
                new_list = list(base_list)
                # Apply the new values to the chosen indices.
                for i, val in enumerate(new_values):
                    index_to_change = indices_to_deviate[i]
                    new_list[index_to_change] = val
                yield new_list


# options = [
#     list(l) for l in itertools.product(*[range(d + 1, 2**d + 1) for d in range(b + 1, -1, -1)])
#     if all(l[i + 1] >= l[i] for i in range(len(l) - 1))
# ]

options = [e + [2**b] for e in efficient_options_generator(b - 1, max_deviations=3)]
np.random.shuffle(options)
# options = [[1, 2, 4, 8, 14, 25, 47, 128]] + options
# options = sorted(options)
print(f'num options = {len(options)}')

m = 2**b - 1
n_range = range(2, min(m - 1, 100), 1)

# Store a tuple (value, source_list) for tracking
best = (float('inf'), None)
best_for_n = {n: (float('inf'), None) for n in n_range}


with mp.Pool(num_cores) as pool:
    task_func = partial(
        combined_worker, b_val=b, restarts_val=restarts, seed_val=seed, m_val=m, n_range_val=n_range
    )

    for max_overlaps_res, result_len, allocation_results in pool.imap_unordered(task_func, options):
        print(f'b = {b} max_overlaps = {max_overlaps_res} len(result) = {result_len}')

        for res in allocation_results:
            n, phi_c, t = res['n'], res['phi_c'], res['t']

            # Update the tuple if a new minimum is found
            if phi_c < best[0]:
                best = (phi_c, max_overlaps_res)

            if phi_c < best_for_n[n][0]:
                best_for_n[n] = (phi_c, max_overlaps_res)

            # MODIFICATION: Updated print statement with full precision and best 'max_overlaps'
            current_best_val, current_best_mo = best
            current_best_for_n_val, current_best_for_n_mo = best_for_n[n]
            print(
                f'n = {n} m = {m} t = {t} phi_c = {phi_c} best = {current_best_val} (mo={current_best_mo}) best_for_n[{n}] = {current_best_for_n_val} (mo={current_best_for_n_mo})'
            )

# MODIFICATION: Final printout with full precision
print("\n--- Final Results ---")
print(f"Overall best phi_c: {best[0]}")
print(f"Found with max_overlaps: {best[1]}")
print("\nBest results per n:")
for n in n_range:
    val, params = best_for_n[n]
    if params:  # Check if a result was ever found for this n
        print(f"  n={n}: phi_c={val}, max_overlaps={params}")
