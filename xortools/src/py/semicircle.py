"""Code for numerical checks of formulas in the proof of theorem 2.1 (the semicircle law).

Coding principles:
1. Prefer correctness by inspection over performance, generality, reuse etc.
2. Prefer similarity to formulas in paper over pythonic idioms.

Example invocation from repo root:

$ pytest src/py/semicircle_test.py -v -n 16

Execution time ~5m.

"""

from typing import Callable, Iterator, List, Set

import itertools
import math
import random

import numpy as np


def generate_all_symbol_strings(
    p: int, length: int, start: int = 0
) -> Iterator[List[int]]:
    r"""Generate all vectors in {start, ..., p - 1}^length."""
    if length == 0:
        yield []
    else:
        for front in range(start, p):
            for tail in generate_all_symbol_strings(p, length - 1, start=start):
                yield [front] + tail


assert list(generate_all_symbol_strings(7, 1, start=1)) == [[x] for x in range(1, 7)]
assert list(generate_all_symbol_strings(3, 2)) == [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 1],
    [1, 2],
    [2, 0],
    [2, 1],
    [2, 2],
]


def generate_symbol_strings_of_fixed_hamming_weight(
    p: int, length: int, weight: int
) -> Iterator[List[int]]:
    for positions in itertools.combinations(range(length), weight):
        g = generate_all_symbol_strings(p, weight, start=1)
        for values in g:
            s = [0] * length
            assert len(positions) == len(values)
            for pos, val in zip(positions, values):
                s[pos] = val
            yield s


assert list(generate_symbol_strings_of_fixed_hamming_weight(3, 4, 0)) == [[0, 0, 0, 0]]
assert list(generate_symbol_strings_of_fixed_hamming_weight(3, 3, 2)) == [
    [1, 1, 0],
    [1, 2, 0],
    [2, 1, 0],
    [2, 2, 0],
    [1, 0, 1],
    [1, 0, 2],
    [2, 0, 1],
    [2, 0, 2],
    [0, 1, 1],
    [0, 1, 2],
    [0, 2, 1],
    [0, 2, 2],
]


def symbol_string_to_index(p: int, s: np.ndarray) -> int:
    """Interpret array as a p-base index with first element most significant."""
    if len(s) == 0:
        return 0
    assert s.dtype == int
    assert 0 <= min(s)
    assert max(s) < p
    return s[-1] + p * symbol_string_to_index(p, s[:-1])


assert symbol_string_to_index(3, np.array([1, 0, 2], dtype=int)) == 11


def hamming_weight(s: np.ndarray) -> int:
    assert s.dtype == int
    hw = 0
    for x in s:
        if x != 0:
            hw += 1
    return hw


assert hamming_weight(np.array([1, 0, 0, 2], dtype=int)) == 2


def make_matrix_A(m: int, ell: int, d: float) -> np.ndarray:
    """Prepare matrix from Lemma 4.2."""
    A = np.zeros((ell + 1, ell + 1), dtype=float)
    for k in range(1, ell + 1):
        v = np.sqrt(k * (m - k + 1))
        A[k - 1, k] = v
        A[k, k - 1] = v
    for k in range(ell + 1):
        A[k, k] = k * d
    return A


assert np.all(
    make_matrix_A(m=9, ell=2, d=7) == np.array([[0, 3, 0], [3, 7, 4], [0, 4, 14]])
)


def kron_pow(m: np.ndarray, n: int) -> np.ndarray:
    mn = np.eye(1)
    for i in range(n):
        mn = np.kron(mn, m)
    return mn


assert np.all(kron_pow(np.eye(4), 3) == np.eye(64))
assert np.all(
    kron_pow(np.reshape(range(4), (2, 2)), 2)
    == np.array([[0, 0, 0, 1], [0, 0, 2, 3], [0, 2, 0, 3], [4, 6, 6, 9]])
)


def omega(p: int) -> complex:
    return np.exp(2j * np.pi / p)


assert abs(omega(2) + 1) < 1e-10
assert abs(omega(3) - (-1 + 1j * np.sqrt(3)) / 2) < 1e-10
assert abs(omega(4) - 1j) < 1e-10


def fourier_transform(p: int, f: Callable[[int], complex]) -> Callable[[int], complex]:
    om = omega(p)

    def f_tilde(y: int) -> complex:
        return sum(f(x) * om ** (x * y) for x in range(p)) / np.sqrt(p)

    return f_tilde


assert (
    abs(fourier_transform(2, lambda x: 2 if x == 0 else 3)(0) - (2 + 3) / np.sqrt(2))
    < 1e-10
)
assert (
    abs(fourier_transform(2, lambda x: 2 if x == 0 else 3)(1) - (2 - 3) / np.sqrt(2))
    < 1e-10
)


def make_single_qudit_fourier_matrix(p: int) -> np.ndarray:
    F = np.zeros((p, p), dtype=complex)
    om = omega(p)
    for i in range(p):
        for j in range(p):
            F[i, j] = om ** (i * j)
    return F / np.sqrt(p)


assert (
    sum(
        sum(
            abs(
                make_single_qudit_fourier_matrix(3)
                @ make_single_qudit_fourier_matrix(3)
                - np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            )
        )
    )
    < 1e-10
)
assert sum(
    sum(
        abs(
            make_single_qudit_fourier_matrix(4)
            - 0.5
            * np.array(
                [[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]]
            )
        )
    )
)


def make_single_qudit_shift_matrix(p: int) -> np.ndarray:
    clock = np.zeros((p, p), dtype=complex)
    for b in range(p):
        clock[(b + 1) % p, b] = 1
    return clock


assert np.all(
    make_single_qudit_shift_matrix(3)
    @ make_single_qudit_shift_matrix(3)
    @ make_single_qudit_shift_matrix(3)
    == np.eye(3)
)
assert np.all(
    make_single_qudit_shift_matrix(5)
    == np.array(
        [
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ]
    )
)


def make_single_qudit_clock_matrix(p: int) -> np.ndarray:
    om = omega(p)
    shift = np.zeros((p, p), dtype=complex)
    for b in range(p):
        shift[b, b] = om**b
    return shift


assert (
    sum(
        sum(
            abs(
                make_single_qudit_clock_matrix(3)
                @ make_single_qudit_clock_matrix(3)
                @ make_single_qudit_clock_matrix(3)
                - np.eye(3)
            )
        )
    )
    < 1e-10
)
assert (
    sum(
        sum(
            abs(
                make_single_qudit_clock_matrix(5)
                - np.diag([1, omega(5), omega(5) ** 2, omega(5) ** 3, omega(5) ** 4])
            )
        )
    )
    < 1e-10
)


def compute_code_distance(p: int, B: np.ndarray) -> int:
    assert B.dtype == int
    m, n = B.shape
    assert m > n
    zero = np.zeros(n, dtype=int)
    for weight in range(1, m + 1):
        # TODO(viathor): Use eigendecomposition of BB^T and/or bounds on code distance to speed this up.
        for candidate in generate_symbol_strings_of_fixed_hamming_weight(p, m, weight):
            y = np.array(candidate, dtype=int)
            if np.all((y.T @ B) % p == zero):
                return weight
    assert False


assert (
    compute_code_distance(
        2,
        np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ]
        ),
    )
    == 3
)
assert (
    compute_code_distance(
        2,
        np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
            ]
        ),
    )
    == 4
)


class MaxLinSat:
    @staticmethod
    def random_instance(p: int, r: int, m: int, n: int) -> "MaxLinSat":
        B = np.random.randint(low=0, high=p, size=(m, n), dtype=int)
        Fs = []
        for i in range(m):
            Fs.append(set(random.sample(range(p), r)))
        return MaxLinSat(p, B, Fs)

    def __init__(self, p: int, B: np.ndarray, Fs: List[Set[int]]) -> None:
        m, n = B.shape

        assert m > n, (m, n)
        assert len(Fs) == m
        r = len(Fs[0])
        for F in Fs:
            assert F.issubset(set(range(p)))
            assert len(F) == r

        self.p = p  # Cardinality of the finite field
        self.r = r  # Cardinality of constraint supports
        self.m = m  # Number of constraints
        self.n = n  # Number of variables
        self.B = B  # Linear transformation from F_p^n to F_p^m
        self.Fs = Fs  # Supports of all constraints
        self.dmin = None  # Minimum code distance of C^\perp

    def __eq__(self, other: "MaxLinSat") -> bool:
        return self.p == other.p and np.all(self.B == other.B) and self.Fs == other.Fs

    def __hash__(self) -> int:
        return hash(
            (
                self.p,
                ((Bij for Bij in bi) for bi in self.B),
                (frozenset(Fi) for Fi in self.Fs),
            )
        )

    def __str__(self) -> str:
        return f"MaxLinSat {self.p=} {self.r=} {self.m=} {self.n=}"

    def get_bi(self, i: int) -> np.ndarray:
        assert 1 <= i <= self.m
        return self.B[i - 1]

    def get_fi(self, i: int) -> Set[int]:
        assert 1 <= i <= self.m
        return self.Fs[i - 1]

    def alpha_i(self, i: int) -> float:
        assert 1 <= i <= self.m
        f_i = self.make_f_i(i)
        return sum(f_i(x) for x in range(self.p)) / self.p

    def make_f_i(self, i: int) -> Callable[[int], int]:
        assert 1 <= i <= self.m, (i, self.m)

        def f_i(x: int) -> int:
            if x in self.get_fi(i):
                return +1
            return -1

        return f_i

    def make_g_i(self, i: int) -> Callable[[int], float]:
        f_i = self.make_f_i(i)
        alpha = self.alpha_i(i)

        def g_i(x: int) -> float:
            return f_i(x) - alpha

        return g_i

    def make_h_i(self, i: int) -> Callable[[int], float]:
        g_i = self.make_g_i(i)
        c = 0.5 * np.sqrt(self.p / self.r / (self.p - self.r))

        def h_i(x: int) -> float:
            return c * g_i(x)

        return h_i

    def make_f_i_tilde(self, i: int) -> Callable[[int], complex]:
        return fourier_transform(self.p, self.make_f_i(i))

    def make_g_i_tilde(self, i: int) -> Callable[[int], complex]:
        return fourier_transform(self.p, self.make_g_i(i))

    def make_h_i_tilde(self, i: int) -> Callable[[int], complex]:
        return fourier_transform(self.p, self.make_h_i(i))

    def make_h_tilde(self) -> Callable[[np.ndarray], complex]:
        def h_tilde(y: np.ndarray):
            assert y.shape == (self.m,)
            assert y.dtype == int
            ret = 1.0 + 0.0j
            for i, yi in enumerate(y):
                if yi != 0:
                    h_i_tilde = self.make_h_i_tilde(i + 1)
                    ret *= h_i_tilde(yi)
            return ret

        return h_tilde

    def make_phi(self, A: Set[int]) -> Callable[[int], complex]:
        om = omega(self.p)

        def phi(y: int) -> complex:
            return sum(om ** (y * u - y * v) for u in A for v in A)

        return phi

    def make_phi_i(self, i: int) -> Callable[[int], complex]:
        return self.make_phi(self.get_fi(i))

    def num_constraints_satisfied(self, x: np.ndarray) -> int:
        """Returns number of constraints satisfied by x."""
        assert x.shape == (self.n,)
        ncs = 0
        for i in range(1, self.m + 1):
            f_i = self.make_f_i(i)
            v = f_i((self.get_bi(i) @ x) % self.p)
            assert v in (-1, +1)
            if v == +1:
                ncs += 1
        return ncs

    def make_sf_observable(self) -> np.ndarray:
        """Returns observable S_f for number of satisfied constraints."""
        s_obs = np.zeros((self.p**self.n, self.p**self.n), dtype=complex)
        for s in generate_all_symbol_strings(self.p, self.n):
            x = np.array(s)
            i = symbol_string_to_index(self.p, x)
            s_obs[i, i] = self.num_constraints_satisfied(x)
        return s_obs

    def make_z_product(self, i: int, a: int) -> np.ndarray:
        """Product of Z_j**(a * B_ij)."""
        z = make_single_qudit_clock_matrix(self.p)
        zn = np.eye(1)
        for j in range(self.n):
            zn = np.kron(zn, np.linalg.matrix_power(z, a * self.get_bi(i)[j]))
        return zn

    def make_x_product(self, i: int, a: int) -> np.ndarray:
        """Product of X_j**(-a * B_ij)."""
        x = make_single_qudit_shift_matrix(self.p)
        xn = np.eye(1)
        for j in range(self.n):
            xn = np.kron(xn, np.linalg.matrix_power(x, -a * self.get_bi(i)[j]))
        return xn

    def compute_code_distance(self) -> int:
        r"""Computes minimum code distance of C^\perp."""
        if self.dmin is None:
            self.dmin = compute_code_distance(self.p, self.B)
        return self.dmin

    def max_ell(self) -> int:
        """Returns maximum ell allowed with this max-LINSAT instance."""
        dmin = self.compute_code_distance()
        return dmin // 2 - 1

    def make_fdqi_state(self, w: np.ndarray) -> np.ndarray:
        """Fourier transformed DQI state."""
        ell = len(w) - 1
        assert 0 <= ell <= self.m / 2
        psi = np.zeros(self.p**self.n, dtype=complex)
        h_tilde = self.make_h_tilde()
        for k in range(ell + 1):
            c = w[k] / np.sqrt(math.comb(self.m, k))
            for s in generate_symbol_strings_of_fixed_hamming_weight(self.p, self.m, k):
                y = np.array(s)
                i = symbol_string_to_index(self.p, (y @ self.B) % self.p)
                assert 0 <= i < len(psi)
                if psi[i] != 0.0:
                    dmin = compute_code_distance(self.p, self.B)
                assert (
                    psi[i] == 0.0
                ), f"Overwriting DQI state amplitude: code distance must be too small ({2*ell+1=}, but {dmin=})"
                psi[i] = c * h_tilde(y)
        return psi
