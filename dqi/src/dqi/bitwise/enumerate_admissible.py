import time

from dqi.bitwise import find_subspace_evasive_set

p = 2
restarts = 10000
seed = 23948573


def is_admissible_pair(n, w1, w2, d):
    if (n - w1 + 1) + (n - w2 + 1) <= n:
        return False, 'is classically easy'

    # This is the minimum size required for the set S_g (g-constraint)
    # to generate the subspace hologram
    N_required_g = p**n - p**d + 1
    S_g = find_subspace_evasive_set(n, w1, restarts, seed, 0)
    assert len(set(S_g)) == len(S_g)
    S_g = sorted(S_g)[:N_required_g]
    if len(S_g) < N_required_g:
        return False, 'not enough elements can fit in S_g'
    # We must contain one element from each coset of V^\perp for
    # the orthogonalising hologram (f-constraint)
    S_f = find_subspace_evasive_set(n, w2, restarts, seed, d)
    assert len(set(S_f)) == len(S_f)
    S_f = sorted(S_f)
    if not len(S_f):
        return False, 'could not hit an element of each coset of V^perp in S_f'
    return True, (S_g, S_f)


for n in range(1, 7):
    for w1 in range(1, n + 1):
        for w2 in range(1, n + 1):
            for d in range(n - 1, 0, -1):
                print(f'{n,w1,w2,d}')
                viable, g_f = is_admissible_pair(n, w1, w2, d)
                if viable:
                    print(f'n={n} w1={w1} w2={w2} d={d} viable')
                    print(f'S_g={g_f[0]}\nS_f={g_f[1]}')
                # else:
                #   print(f'n={n} w1={w1} w2={w2} d={d} non-viable because {g_f}')
