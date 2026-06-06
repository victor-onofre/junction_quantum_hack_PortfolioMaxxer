import galois
import numpy as np
from galois import Poly
from qualtran.bloqs.swap_network.swap_with_zero import _swap_with_zero_swap_sequence


def poly_eea_zalka_gcd(A: Poly, B: Poly) -> Poly:
    gf = A.field
    one, zero = Poly.One(gf), Poly.Zero(gf)
    if A.degree > B.degree:
        A, B = B, A
    a, b = one, zero
    m = B.degree
    while A != zero:
        assert A.degree <= B.degree
        assert (a * B).degree == m
        assert a.degree + A.degree <= m and b.degree + B.degree <= m
        q = B // A
        assert q.degree == B.degree - A.degree
        assert galois.gcd(A, B) == galois.gcd(B - q * A, A)
        B = B - q * A
        assert B.degree < A.degree or B == zero
        b = b - q * a
        assert (A * b).degree == m
        a, b = b, a
        A, B = B, A
        assert q == -a // b
    return B // B.coeffs[0]


def poly_divmod(A: Poly, B: Poly) -> tuple[Poly, Poly, Poly]:
    r"""Reversibly compute (A, B) --> (A, B // A, B % A)

    B = qA + r

    deg(B) = deg(q) + deg(A) and deg(r) < deg(A)

    n_terms(q) = deg(q) + 1 = deg(B) - deg(A) + 1
    n_terms(r) = deg(r) + 1 <= deg(A)
    n_terms(q + r) <= deg(B) + 1

    Poly Div Mod is in-place for `B` register!
    """
    assert A.degree <= B.degree
    (expected_q, expected_r) = (B // A, B % A)
    n_iters = B.degree - A.degree + 1
    # Normalize A such that max degree coefficient is 1.
    a_inv = A.coeffs[0] ** -1
    x = Poly.Degrees(coeffs=[1], degrees=[1], field=A.field)
    b_degree = B.degree
    q_coeffs = A.field.Zeros(n_iters)
    for i in range(n_iters):
        q_coeffs[i] = B.coefficients(b_degree + 1 - i)[0] * a_inv
        B -= A * q_coeffs[i] * x ** (n_iters - i - 1)
    q = Poly(q_coeffs, field=A.field)
    assert (q, B) == (expected_q, expected_r)
    return (A, q, B)


def poly_divmod_inv(A: Poly, q: Poly, r: Poly) -> tuple[Poly, Poly]:
    r"""Reversibly compute (A, q, r) --> (A, qA + r) where deg(r) < deg(A).

    B = qA + r

    deg(B) = deg(q) + deg(A) and deg(r) < deg(A)

    n_terms(q) = deg(q) + 1 = deg(B) - deg(A) + 1
    n_terms(r) = deg(r) + 1 <= deg(A)
    n_terms(q + r) <= deg(B) + 1

    Poly Div Mod is in-place for `B` register!
    """
    assert r.degree <= A.degree
    n_iters = q.degree + 1
    q_coeffs = q.coeffs
    x = Poly.Degrees(coeffs=[1], degrees=[1], field=A.field)
    assert len(q_coeffs) == n_iters
    for i in range(n_iters):
        r += A * q_coeffs[-i - 1] * x**i
    return A, r


def poly_eea_zalka_lfsr(s: galois.FieldArray) -> galois.FieldArray:
    # Step-1: Setup EEA polynomials.
    gf = type(s[0])
    A = Poly(s[::-1], field=gf)
    B = Poly.Degrees(coeffs=[1], degrees=[len(s)], field=gf)
    # Step-2: Solve EEA
    one, zero = Poly.One(gf), Poly.Zero(gf)
    a, b = one, zero
    m = B.degree
    while A != zero and B.degree > b.degree:
        assert A.degree < B.degree
        assert (a.degree > b.degree) or (a == one and b == zero)
        assert (a * B).degree == m
        assert a.degree + A.degree <= m and b.degree + B.degree <= m
        A, q, B = poly_divmod(A, B)
        assert q.degree <= np.ceil(3 * np.log2(m))
        a, b = poly_divmod_inv(a, q, b)
        # Swap!
        A, B = B, A
        a, b = b, a
    return (b // (gf(0) - b.coeffs[-1])).coeffs[::-1]


def shift_array_right(a_array: np.ndarray, deg_A: int):
    for i in range(1, deg_A + 1):
        a_array[-(i + 1)], a_array[-i] = a_array[-i], a_array[-(i + 1)]


def _to_poly(coeffs: np.ndarray, field: galois.FieldArray) -> Poly:
    return Poly.Zero(field) if not len(coeffs) else Poly(coeffs, field=field)


def _poly_eea_zalka_gcd_helper(A: Poly, B: Poly, stop, n_iters: int) -> tuple[np.ndarray, int, int]:
    r"""Zalka EEA implemented using register sharing and synchronized reversible steps.

    To compute gcd(A, B); we maintain (a, A) and (b, B) such that
        - deg(A) < deg(B) and deg(a) > deg(b)
        - deg(aB) = m
        - deg(a) + deg(A) <= m and deg(b) + deg(B) <= m

    Using the above facts, we store (a, A) and (b, B) in a single shared register
    of length `m` each. Thus, the total space needed is `2m`.
        - (a, A): [a_{m1}, a_{m1-1}, ..., a_0, A_0, A_1, ..., A_{m2}]
        - (b, B): [b_{n1}, b_{n1-1}, ..., b_0, B_0, B_1, ..., B_{n2}]
        - q : [q_{d}, q_{d - 1}, ..., q_{0}]

    In each step, we want to replace (a, A), (b, B) with
    (qa - b, B - qA), (a, A). To do this reversibly, we let the computation
    in each branch of superposition proceed at its own pace such that in each
    step; we apply one of the following four operations:
        - o1: Compute A, q, B = poly_divmod(A, B) where B may have trailing zeros.
        - o2: Remove trailing zeroes from B
        - o4: Shift `b` right by `n_a - n_b` steps
        - o3: Compute a, b = poly_divmod_inv(a, q, b) in-place.

    This is an optimized implementation compared to the original approach, where
    we merge steps o1 and o3 and o2 and o4 to arrive at a unified cycle architecture.

    The shared registers for (a, A) and (b, B) have the following structure:
        - a_array: a_{na}, a_{na-1}, ..., a_{0} | 0, 0, ..., 0, A_0, A_1, ..., A_{nA}
        - b_array: b_{nb}, ..., b_{0}, 0,..., 0 | B_0, B_1, ................., B_{nB}
    """
    gf = A.field
    m = B.degree + 3
    a_array, b_array = gf.Zeros(m), gf.Zeros(m)
    a_array[0] = 1
    a_array[m - A.degree - 1 :] = A.coeffs[::-1]
    b_array[m - B.degree - 1 :] = B.coeffs[::-1]
    n_A, n_B, n_a, n_b, n_q = A.degree + 1, B.degree + 1, 1, 0, 0
    flag, counter, halting_counter = 1, 0, 0

    def swap_a_and_b():
        nonlocal n_A, n_B, flag, n_a, n_b, a_array, b_array, halting_counter, counter
        if counter == 3 and flag and halting_counter == 0:
            n_A, n_B = n_B, n_A
            n_a, n_b = n_b, n_a
            a_array, b_array = b_array, a_array

    def update_flag():
        nonlocal n_A, n_B, n_q, flag, n_a, n_b, a_array, b_array
        if counter == 0:
            is_first = (n_q == 0) and (n_A < n_B)
            is_last = n_A == n_B
        elif counter == 1:
            is_first = n_A == n_B
            is_last = n_B == 1 or b_array[-2] != 0
        elif counter == 2:
            is_first = b_array[0] != 0 or n_b == 0
            is_last = n_a == n_b + 1
        elif counter == 3:
            is_first = n_a == n_b
            is_last = n_q == 1
        else:
            assert False
        if halting_counter == 0:
            flag ^= is_first ^ is_last

    def multiply_and_add_or_sub():
        # N CSWAPS, N CADDS, N Multiplications
        q_i = 0
        if counter == 0 and halting_counter == 0:
            q_i ^= (a_array[-1] ** -1) * b_array[-1]

        for i in range(m - 1, -1, -1):
            swap_if = i == n_a + n_q and counter == 0 and halting_counter == 0
            swap_if |= i + 1 == n_a + n_q and counter == 3 and halting_counter == 0
            if swap_if:
                q_i, a_array[i] = a_array[i], q_i

            q_i_times_a_i = q_i * a_array[i]
            add_if = (counter == 3) and i < n_a
            sub_if = (counter == 0) and i >= m - n_A
            if add_if:
                b_array[i] += q_i_times_a_i
            if sub_if:
                b_array[i] -= q_i_times_a_i

        if counter == 3 and halting_counter == 0:
            q_i ^= b_array[0] * (a_array[0] ** -1)
        assert q_i == 0

    def shift_b_array_right():
        nonlocal b_array
        ctrl_n_B = (counter == 1) | (counter == 0 and not flag)
        for i in range(m - 1, -1, -1):
            do_shift = False
            do_shift ^= (n_B >= m - i + 1) and ctrl_n_B
            do_shift ^= (0 < i <= n_b) and (counter == 2)
            do_shift ^= (0 < i < m - n_B) and not flag and counter == 3
            if do_shift and halting_counter == 0:
                b_array[i - 1], b_array[i] = b_array[i], b_array[i - 1]

    def update_metadata():
        nonlocal n_A, n_B, n_q, flag, n_a, n_b

        if halting_counter != 0:
            return
        if counter == 0:
            n_q += 1
            if not flag:
                # This is not the last step!
                n_B -= 1
        if counter == 1:
            n_B -= 1
        if counter == 2:
            n_b += 1
        if counter == 3:
            if not flag:
                n_b += 1
            n_q -= 1

    def check_stop():
        nonlocal halting_counter
        if counter == 3 and flag and stop(n_A, n_B, n_a, n_b):
            halting_counter += 1

    for it in range(n_iters):
        # Totals:
        #   - 3N CSWAPS
        #   - N Multiplications
        #   - N Controlled Additions / Subtractions
        #   - 4Nb + N * b^log_3{2}
        update_flag()  # ~0
        multiply_and_add_or_sub()  # N Multiplications, N CADDS, N CSWAPS
        shift_b_array_right()  # N CSWAPS
        update_metadata()  # ~0
        swap_a_and_b()  # N CSWAPS
        check_stop()  # ~0
        # Advance counter.
        counter = (counter + flag) % 4
    assert halting_counter > 0
    return b_array, n_b, n_B


def _poly_eea_zalka_gcd_helper_slow(
    A: Poly, B: Poly, stop, n_iters: int
) -> tuple[np.ndarray, int, int]:
    r"""Zalka EEA implemented using register sharing and synchronized reversible steps.

    To compute gcd(A, B); we maintain (a, A) and (b, B) such that
        - deg(A) < deg(B) and deg(a) > deg(b)
        - deg(aB) = m
        - deg(a) + deg(A) <= m and deg(b) + deg(B) <= m

    Using the above facts, we store (a, A) and (b, B) in a single shared register
    of length `m` each. Thus, the total space needed is `2m`.
        - (a, A): [a_{m1}, a_{m1-1}, ..., a_0, A_0, A_1, ..., A_{m2}]
        - (b, B): [b_{n1}, b_{n1-1}, ..., b_0, B_0, B_1, ..., A_{n2}]
        - q : [q_{d}, q_{d - 1}, ..., q_{0}]

    In each step, we want to replace (a, A), (b, B) with
    (qa - b, B - qA), (a, A). To do this reversibly, we let the computation
    in each branch of superposition proceed at its own pace such that in each
    step; we apply one of the following four operations:
        - o1: Compute A, q, B = poly_divmod(A, B) where B may have trailing zeros.
        - o2: Remove trailing zeroes from B
        - o3: Shift `b` right by `n_a - n_b` steps
        - o4: Compute a, b = poly_divmod_inv(a, q, b) in-place.

    The shared registers for (a, A) and (b, B) have the following structure:
        - a_array: a_{na}, a_{na-1}, ..., a_{0} | 0, 0, ..., 0, A_0, A_1, ..., A_{nA}
        - b_array: b_{nb}, ..., b_{0}, 0,..., 0 | B_0, B_1, ................., B_{nB}
    """
    gf = A.field
    m = B.degree + 3
    log_m = m.bit_length()
    a_array, b_array = gf.Zeros(m), gf.Zeros(m)
    a_array[0] = 1
    a_array[m - A.degree - 1 :] = A.coeffs[::-1]
    b_array[m - B.degree - 1 :] = B.coeffs[::-1]
    n_A, n_B, n_a, n_b, n_q = A.degree + 1, B.degree + 1, 1, 0, 0
    flag, counter, halting_counter = 1, 0, 0

    def apply_o0():
        r"""Compute A, q, B = poly_divmod(A, B)"""
        # N controlled subtractions
        # 3N CSWAPS
        # N quantum-quantum multiplications
        nonlocal n_B, n_q, flag, n_a, n_b
        if counter != 0 or halting_counter != 0:
            return
        is_first = (n_q == 0) and (n_A < n_B)
        is_last = n_A == n_B
        flag ^= is_first ^ is_last
        assert a_array[n_a + n_q] == 0
        q_i = (a_array[-1] ** -1) * b_array[-1]
        for i in range(1, m):
            q_i_times_a_i = q_i * a_array[-i]
            if i <= n_A:
                b_array[-i] -= q_i_times_a_i
            q_i_times_a_i ^= q_i * a_array[-i]
            assert q_i_times_a_i == 0

        # Perform a_array[n_a + n_q] = q_i using a SwapWithZero
        idx_to_set = n_a + n_q
        swap_sequence = [*_swap_with_zero_swap_sequence((log_m,), (m,))]
        assert len(swap_sequence) == m - 1
        for _, ctrl_idx, i, j in swap_sequence:
            if (1 << ctrl_idx) & idx_to_set:
                a_array[i], a_array[j] = a_array[j], a_array[i]
        assert a_array[0] == 0
        a_array[0], q_i = q_i, a_array[0]
        assert q_i == 0
        for _, ctrl_idx, i, j in swap_sequence[::-1]:
            if (1 << ctrl_idx) & idx_to_set:
                a_array[i], a_array[j] = a_array[j], a_array[i]
        # Increment n_q
        n_q += 1
        if not flag:
            # This is not the last step!
            n_B -= 1
        for i in range(m - 1, -1, -1):
            if i >= m - n_B and not flag:
                b_array[i - 1], b_array[i] = b_array[i], b_array[i - 1]

    def apply_o1():
        r"""Right shift B until b_array[-1] != 0 and update n_B."""
        # N CSWAPs
        nonlocal n_B, flag
        if counter != 1 or halting_counter != 0:
            return
        is_first = n_A == n_B
        is_last = n_B == 1 or b_array[-2] != 0
        flag ^= is_first ^ is_last
        assert b_array[-1] == 0
        n_B -= 1
        for i in range(m - 1, -1, -1):
            if i >= m - n_B:
                b_array[i - 1], b_array[i] = b_array[i], b_array[i - 1]

    def apply_o2():
        r"""Right shift B s.t. b and a share the boundary."""
        # N CSWAPS
        nonlocal n_A, n_B, a_array, b_array, n_a, n_b, flag
        if counter != 2 or halting_counter != 0:
            return
        assert (n_A > n_B) and (n_a > n_b)
        is_first = b_array[0] != 0 or n_b == 0
        is_last = n_a == n_b + 1
        flag ^= is_first ^ is_last

        for i in range(m - 1, -1, -1):
            if i < n_b:
                b_array[i], b_array[i + 1] = b_array[i + 1], b_array[i]
        n_b += 1

    def apply_o3():
        r"""Compute a, b = poly_divmod_inv(a, q, b) = (a, qa + b)"""
        # N controlled subtractions
        # 3N CSWAPS
        # N quantum-quantum multiplications
        nonlocal n_B, n_q, flag, n_a, n_b
        if counter != 3 or halting_counter != 0:
            return
        assert n_A > n_B
        is_first = n_a == n_b
        is_last = n_q == 1

        flag ^= is_first ^ is_last

        n_q -= 1
        assert b_array[0] == 0 and a_array[0] != 0
        # Perform q_i = a_array[n_a + n_q] using a SwapWithZero
        q_i = 0
        idx_to_set = n_a + n_q
        swap_sequence = [*_swap_with_zero_swap_sequence((log_m,), (m,))]
        assert len(swap_sequence) == m - 1
        for _, ctrl_idx, i, j in swap_sequence:
            if (1 << ctrl_idx) & idx_to_set:
                a_array[i], a_array[j] = a_array[j], a_array[i]
        a_array[0], q_i = q_i, a_array[0]
        assert a_array[0] == 0
        for _, ctrl_idx, i, j in swap_sequence[::-1]:
            if (1 << ctrl_idx) & idx_to_set:
                a_array[i], a_array[j] = a_array[j], a_array[i]
        assert a_array[n_a + n_q] == 0
        # Done.
        for i in range(m):
            q_i_times_a_i = q_i * a_array[i]
            if i < n_a:
                b_array[i] += q_i_times_a_i
        q_i ^= b_array[0] * (a_array[0] ** -1)
        assert q_i == 0, f'{q_i=}, {n_q=}, {n_a=}'
        if not flag:
            # Not the last step; shift the b_array right in prep for next addition.
            for i in range(m - 1, -1, -1):
                if i < m - n_B - 1:
                    b_array[i], b_array[i + 1] = b_array[i + 1], b_array[i]
            n_b += 1
            assert b_array[0] == 0

    def swap_a_and_b():
        r"""Right shift a s.t. a and A share the boundary."""
        # N CSWAPS
        nonlocal n_A, n_B, a_array, b_array, n_a, n_b, flag
        if counter == 3 and halting_counter == 0 and flag == 1:
            assert (n_A > n_B) and n_q == 0 and (n_a < n_b)
            # Swap a and b
            n_A, n_B = n_B, n_A
            n_a, n_b = n_b, n_a
            a_array, b_array = b_array, a_array

    steps = [apply_o0, apply_o1, apply_o2, apply_o3]
    # print(f'{a_array=}')
    # print(f'{b_array=}')
    # print(f'{n_a=}, {n_b=}, {n_A=}, {n_B=}, {n_q=}')
    for it in range(n_iters):
        for step in steps[::-1]:
            step()
        swap_a_and_b()
        if counter == 3 and flag and stop(n_A, n_B, n_a, n_b):
            halting_counter += 1
        # print(f'{a_array=}')
        # print(f'{b_array=}')
        # print(f'{n_a=}, {n_b=}, {n_A=}, {n_B=}, {n_q=}, {halting_counter=}')
        # assert (
        #     n_a + n_B == m - 1 if counter == 3 else True
        # ), f'{n_a=}, {n_B=}, {m=}, {it=}, {counter=}'
        # Advance counter.
        counter = (counter + flag) % len(steps)
    return b_array, n_b, n_B


def poly_eea_zalka_gcd_sync(A: Poly, B: Poly) -> Poly:
    if A.degree > B.degree:
        A, B = B, A

    def _stop_gcd(n_A: int, _: int, __: int, ___: int):
        return n_A == 0

    m, gf = B.degree + 2, B.field
    n_iters = 6 * B.degree
    b_array, n_b, n_B = _poly_eea_zalka_gcd_helper(A, B, _stop_gcd, n_iters=n_iters)
    b = _to_poly(b_array[-n_B:][::-1], gf)
    return b // b.coeffs[0]


def poly_eea_zalka_lfsr_sync(s: galois.FieldArray) -> galois.FieldArray:
    def _stop_lfsr(n_A: int, n_B: int, _: int, n_b: int):
        return n_A == 0 or n_B <= n_b

    gf = type(s[0])
    A = Poly(s[::-1], field=gf)
    B = Poly.Degrees(coeffs=[1], degrees=[len(s)], field=gf)
    n_iters = (3 * (len(s) + 1) + 5) + 3
    b_array, n_b, n_B = _poly_eea_zalka_gcd_helper(A, B, stop=_stop_lfsr, n_iters=n_iters)
    b = _to_poly(b_array[:n_b], gf)
    return (b // (gf(0) - b.coeffs[-1])).coeffs[::-1]
