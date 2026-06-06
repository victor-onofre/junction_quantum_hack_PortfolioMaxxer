import numpy as np
from sympy import symbols, Poly, binomial
from scipy.special import comb
import sympy
from sympy.functions.combinatorial.factorials import binomial
from functools import lru_cache
from gmpy2 import mpz, bincoef
from concurrent.futures import ProcessPoolExecutor
import sys

sys.set_int_max_str_digits(100000)

# Define the polynomial ring variables
U = symbols("U")
S, T = symbols("S T")


def get_first_poly(Lam):
    """
    Compute the first polynomial term based on the left degree distribution Lam.
    Args:
        Lam (list): Left degree distribution (number of variable nodes of each degree).

    Returns:
        Poly: The resulting polynomial.
    """
    result = Poly(1, S, T)
    for i, count in enumerate(Lam):
        if count > 0:
            result *= Poly((1 + T * S**i) ** count, S, T)
    return result


def get_second_poly(Rho):
    """
    Compute the second polynomial term based on the right degree distribution Rho.
    Args:
        Rho (list): Right degree distribution (number of check nodes of each degree).

    Returns:
        Poly: The resulting polynomial.
    """
    result = Poly(1, U)
    for j, count in enumerate(Rho):
        if count > 0:
            term = Poly(((1 + U) ** j + (1 - U) ** j) / 2, U)
            result *= term**count
    return result


def get_e_num_codewords_dist(m, n, Lam, Rho):
    """
    Compute the distribution of expected number of codewords for all weights in the irregular case.
    Args:
        m (int): Number of left-hand (variable) nodes.
        n (int): Number of right-hand (check) nodes.
        Lam (list): Left degree distribution.
        Rho (list): Right degree distribution.

    Returns:
        list: Distribution of expected codewords, where the i-th entry is the expected number
              of codewords of weight i.
    """
    # Sanity checks
    if sum(Lam) != m or sum(Rho) != n:
        raise ValueError("Degree distributions don't match the number of nodes!")
    if sum(i * Lam[i] for i in range(len(Lam))) != sum(
        j * Rho[j] for j in range(len(Rho))
    ):
        raise ValueError("Degree distributions don't match the number of edges!")

    # Compute the first and second polynomials
    FP = get_first_poly(Lam)
    SP = get_second_poly(Rho)

    # Precompute coefficients of SP
    sp_coeffs = SP.as_dict()

    # Precompute coefficients of FP
    fp_coeffs = FP.as_dict()

    # Initialize the distribution
    dist = [0] * (m + 1)

    n_edges = sum(count * i for i, count in enumerate(Lam))

    # Iterate over all k and compute contributions
    kmax = FP.degree(S) + 1
    for (k,), coeff2 in sp_coeffs.items():
        denom = binomial(n_edges, k)
        for ell in range(m + 1):
            coeff1 = fp_coeffs.get(
                (
                    k,
                    ell,
                ),
                0,
            )
            dist[ell] += coeff1 * coeff2 / denom

    return dist


@lru_cache(maxsize=None)
def symbinomial(n, k):
    return binomial(n, k)


# https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes5a.pdf
@lru_cache(maxsize=None)
def krawtchouk(w, t, m):
    """
    Evaluate the w-th Krawtchouk polynomial at t for length m.
    """
    total = 0
    sign = 1
    for j in range(w + 1):
        total += sign * symbinomial(t, j) * symbinomial(m - t, w - j)
        sign *= -1
    return total


def krawtchouk_gmpy2(w, t, m):
    """
    Evaluate the w-th Krawtchouk polynomial at t for length m
    using GMPY2 for arbitrary-precision integer arithmetic.
    """
    # cdef int j
    # cdef mpz result = mpz(0)
    result = mpz(0)
    # cdef mpz sign, binom1, binom2

    sign = mpz(1)
    for j in range(w + 1):
        # Binomial coefficients
        binom1 = bincoef(t, j)
        binom2 = bincoef(m - t, w - j)

        # Add to result
        result += sign * binom1 * binom2

        # Compute (-1)^j
        sign *= mpz(-1)

    return result


def dual_level_weight_times_code_size(arg):
    (w, m, dist) = arg
    return sum(dist[t] * krawtchouk(w, t, m) for t in range(m + 1))


def macwilliams_transform_krawtchouk(m, n, dist):
    """
    Compute the dual code weight distribution using Krawtchouk polynomials.

    Args:
        m (int): Length of the code.
        n (int): Number of parity checks (dimension is m-n).
        dist (list or array): Weight distribution of the original code,
                              where dist[t] is the number of codewords of weight t.

    Returns:
        list: Weight distribution of the dual code.
    """
    # Size of the code
    code_size = sympy.Rational(2) ** (m - n)

    # Initialize the dual weight distribution
    dual_dist = [sympy.Rational(0)] * (m + 1)

    # Compute the dual distribution
    dual_dist = list(
        map(dual_level_weight_times_code_size, [(w, m, dist) for w in range(m + 1)])
    )
    dual_dist = [_ / code_size for _ in dual_dist]
    return dual_dist


def kXORSAT_objective_dist(k, m, n):
    Lam = [0] * k + [m]
    var_degs = np.zeros(n, dtype=int)
    for i in range(m):
        for j in np.random.choice(np.arange(n), size=k, replace=False):
            var_degs[j] += 1
    # Use np.unique to create the Rho array where Rho[i] is the number of entries of var_degs equal to i
    degs, counts = np.unique(var_degs, return_counts=True)
    Rho = np.zeros(max(degs) + 1, dtype=int)
    Rho[degs] = counts

    # Compute the distribution
    dist = get_e_num_codewords_dist(m, n, Lam, Rho)
    dual_dist = macwilliams_transform_krawtchouk(m, n, dist)
    return dual_dist[::-1]


def kXORSAT_ldpc_weight_dist(k, m, n):
    Lam = [0] * k + [m]
    var_degs = np.zeros(n, dtype=int)
    for i in range(m):
        for j in np.random.choice(np.arange(n), size=k, replace=False):
            var_degs[j] += 1
    # Use np.unique to create the Rho array where Rho[i] is the number of entries of var_degs equal to i
    degs, counts = np.unique(var_degs, return_counts=True)
    Rho = np.zeros(max(degs) + 1, dtype=int)
    Rho[degs] = counts

    # Compute the distribution
    dist = get_e_num_codewords_dist(m, n, Lam, Rho)
    return dist


def kXORSAT_reg_ldpc_weight_dist(k, m, n):
    D = (k * m) // n
    print(k, D)
    Lam = [0] * k + [m]
    Rho = [0] * D + [n]
    # Compute the distribution
    dist = get_e_num_codewords_dist(m, n, Lam, Rho)
    return dist


def load_dist(fname):
    dist = []
    with open(fname, "r") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                break
            dist.append(sympy.Rational(line))
    return dist


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "generate expected weight distribution of LDPC and LDGM code"
    )
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    # parser.add_argument("--dps", type=int, required=False, default=10)
    parser.add_argument("--ldpc-infile", type=str, default=None)
    parser.add_argument("--ldpc-outfile", type=str, default=None)
    parser.add_argument("--regular", action="store_true", default=False)
    args = parser.parse_args()
    if args.ldpc_infile is None:
        if args.regular:
            ldpc_dist = kXORSAT_reg_ldpc_weight_dist(args.k, args.m, args.n)
        else:
            ldpc_dist = kXORSAT_ldpc_weight_dist(args.k, args.m, args.n)
        # print([round(e) for e in ldpc_dist])
        if args.ldpc_outfile is not None:
            with open(args.ldpc_outfile, "w") as outfile:
                for num in ldpc_dist:
                    outfile.write(str(num))
                    # outfile.write(str(round(num, 10)))
                    # outfile.write(f"{float(num):.20f}")
                    outfile.write("\n")
    else:
        ldpc_dist = load_dist(args.ldpc_infile)
    # ldgm_dist = macwilliams_transform_krawtchouk(args.m, args.n, ldpc_dist)
    # assert all(e >= 0 for e in ldgm_dist)
    # print([round(e) for e in ldgm_dist])
