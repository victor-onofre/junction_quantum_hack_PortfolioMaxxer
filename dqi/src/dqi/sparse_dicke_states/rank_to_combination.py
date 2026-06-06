import math


def compute_hypergeometric_ps(n1: int, n2: int, k: int, m: int) -> int:
    r"""Computes the Hypergeometric CDF prefix sum PS(m) using Binary Splitting.

    We want to compute the sum $PS(m) = sum_{i=0}^{m-1} C(n1, i) * C(n2, k-i)$,
    which can be written as $PS(m) = T_{0} (1 + R_0 + R_0 R_1 + ...)$
    where $R_{i} = T_{i + 1} / T_{i} = R(i) = \frac{(n1-i)(k-i)}{(i+1)(n2 - k + i + 1)}$
    is an O(log(n)) bit fraction.
    """

    def pq(i):
        """Returns the reduced fraction p'(i), q'(i) for R(i) = p(i) / q(i)."""
        # Handle boundaries where T_{i+1} is 0, so R(i)=0.
        if i >= n1 or i >= k:
            return 0, 1
        return (n1 - i) * (k - i), (i + 1) * (n2 - k + 1 + i)

    def binary_splitting(a, b):
        """
        Computes P(a,b), Q(a,b), S(a,b) for the index range [a, b).
        P/Q is the product of ratios R(a)...R(b-1).
        S/Q is the relative sum S_rel(a,b) = 1 + R(a) + R(a)R(a+1) + ...
        """
        if b - a == 1:
            P, Q = pq(a)
            return P, Q, Q  # S_rel = 1

        mid = (a + b) // 2
        Pl, Ql, Sl = binary_splitting(a, mid)
        Pr, Qr, Sr = binary_splitting(mid, b)

        # This is where the fast l  arge integer multiplication happens.
        return Pl * Pr, Ql * Qr, Sl * Qr + Pl * Sr

    st = max(0, k - n2)
    en = min(m, min(k, n1) + 1)
    if en <= st:
        return 0
    _, Q_tot, S_tot = binary_splitting(st, en)
    T_start = math.comb(n1, st) * math.comb(n2, k - st)
    return int((T_start * S_tot) // Q_tot)


def rank_to_combination_divide_and_conquer(n: int, k: int, r: int) -> tuple:
    r"""Returns the r'th combination among all (n choose k) combinations in O(n polylog(n))
    Uses divide and conquer to get T(n) = 2T(n / 2) + O(n log^2(n)) (assuming k = O(n))
    which gives T(n) = O(n.polylog(n))
    """
    if k == 0 or k == n:
        return tuple([*range(n)]) if k else ()

    n1, n2 = n // 2, (n - n // 2)
    k_left = 0
    for j in range(n.bit_length() - 1, -1, -1):  # O(log(n))
        ps_i = compute_hypergeometric_ps(n1, n2, k, k_left | (1 << j))  # O(k log(n))
        if k_left | (1 << j) <= k and ps_i <= r:  # O(k log(n))
            k_left |= 1 << j

    r -= compute_hypergeometric_ps(n1, n2, k, k_left)  # O(k log(n))
    n2_comb = math.comb(n2, k - k_left)  # O(k log(n))
    r_left, r_right = r // n2_comb, r % n2_comb  # O(k log(n))

    lcomb = rank_to_combination_divide_and_conquer(n1, k_left, r_left)
    rcomb = rank_to_combination_divide_and_conquer(n2, k - k_left, r_right)
    return lcomb + tuple(x + n1 for x in rcomb)


def rank_to_combination(n: int, k: int, r: int) -> tuple:
    r"""Returns the r'th combination among all (n choose k) combinations.

    Toffoli cost: m . k + (k log2(m))^2
    # b * (b * k + b * (k - 1) + ... b * (1))
    # b^2 (1 + 2 + ... + k) = k (k - 1)/2  * b^2
    """
    ret = [0 for _ in range(k)]
    for idx in range(k, 0, -1):
        best_cj = 0
        # The entire loop below corresponds to a single QROM on `b=log2(m)`
        # selection bits with data loading calls attached to not just the `2^b` leaf
        # nodes, but also `2^b - 1` internal nodes. Total Toffoli cost for all
        # sparse QROMs combined is exactly `m=2^b`.
        for j in range(n.bit_length(), -1, -1):  # Runs `log2(m)` times
            # Sparse QROM to compute the binomial coefficient.
            # quantum-quantum comparator on `k . b` bits.
            should_set = math.comb(best_cj + (1 << j), idx) <= r
            if should_set:
                best_cj |= 1 << j
            assert should_set == (best_cj >> j) & 1
        ret[k - idx] = best_cj
        r -= math.comb(best_cj, idx)
    return tuple(ret)
