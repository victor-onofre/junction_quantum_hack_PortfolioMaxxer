import numpy as np
import galois
from galois import Poly
from dqi.gcd.poly_eea_zalka import _poly_eea_zalka_gcd_helper


def get_delta(dialogue: list):
    delta = 0
    for c in dialogue:
        is_swap = delta >= 0 and c != 0
        if is_swap:
            delta = ~delta
        delta = 1 + delta
    return delta


def construct_dialog(s: np.ndarray, n_iters: int) -> tuple[np.ndarray, list, int]:
    gf, n = type(s[0]), len(s)
    poly = [gf(0) for _ in range(2 * n + 3)]
    poly[0], poly[-n:] = gf(1), s
    delta = 0
    dialog = []
    for _ in range(n_iters):
        # 1. Compute the dialog
        is_swap = delta >= 0 and poly[-1] != 0
        if is_swap:
            delta, poly = ~delta, poly[::-1]
        delta = 1 + delta
        coeff = poly[-1] // poly[0]
        dialog.append((coeff, is_swap))
        # 2. Perform omega_x -= coeff * u_x in shared register representation.
        for j in range(len(poly) // 2 - 1, -1, -1):
            if j < len(poly) // 2 + delta // 2:
                poly[-j - 1] -= poly[j] * coeff
        assert poly.pop() == 0

    assert len(poly) == 2 * n + 3 - n_iters
    return poly, dialog, delta


def rs_syndrome_decoder_eea_zalka(s: np.ndarray) -> np.ndarray:
    gf, n = type(s[0]), len(s)

    def _stop_rs(n_A: int, n_B: int, _: int, __: int):
        return n_A == 0 or n_B < n // 2

    A = Poly(s[::-1], field=gf)
    B = Poly.Degrees(coeffs=[1], degrees=[n], field=gf)
    n_iters = (3 * (n + 1) + 5) + 4
    b_array, n_b, n_B = _poly_eea_zalka_gcd_helper(A, B, stop=_stop_rs, n_iters=n_iters)
    assert n_B <= n // 2 and n_b <= n // 2, f'{n=}, {n_B=}, {n_b=}'

    def poly_eval(c):
        # Evaluate omega using shared register representation.
        # b_array = [sigma_{n_b}, ..., sigma{0} | 0 | omega_{0}, ..., omega_{n_B-1}]
        omega_x_val, sigma_x_val, sigma_x_prime_val = gf(0), gf(0), gf(0)
        m = len(b_array)
        for i in range(m - 1, m // 2, -1):
            const_to_mul = c ** (i)
            omega_x_val += b_array[i] * const_to_mul
        for i in range(m // 2, -1, -1):
            const_to_mul = c ** (-i)
            sigma_x_val += b_array[i] * const_to_mul
            if i & 1:
                sigma_x_prime_val += b_array[i] * const_to_mul
        return omega_x_val, sigma_x_val, sigma_x_prime_val

    block_length = gf.order - 1
    gammas = gf([gf.primitive_element**i for i in range(block_length)])
    errors = gf.Zeros(block_length)
    for i in range(block_length):
        omega_x_eval, sigma_x_eval, sigma_x_prime_eval = poly_eval(gammas[i])
        if sigma_x_eval == 0:
            c = gammas[i]
            const_to_mul = c ** -(len(b_array) - 1 + n_b - n_B - 1)
            j = block_length - i if i else i
            errors[j] = const_to_mul * omega_x_eval / sigma_x_prime_eval
    return errors


def rs_syndrome_decoder_eea_divstep_opt(s: np.ndarray) -> np.ndarray:
    gf, n = type(s[0]), len(s)
    # Compute the dialog representation using Bernstein-Yang style GCD iterations.
    # Shared register stores (u_x, omega_x) such that:
    # u_x_{0}, u_x_{1}, ..., u_x_{n}, 0, 0, omega_{n - 1}, omega_{n - 2}, ..., omega_0
    poly, dialog, delta = construct_dialog(s, n - 1)

    def omega_eval(c):
        # Evaluate omega using shared register representation. n_omega = n//2 - delta // 2
        res = gf(0)
        cpow = 1
        for j in range(n // 2):
            if j < n // 2 - delta // 2:
                res += poly[-j - 1] * cpow
            cpow = cpow * c
        return res

    def sigma_and_sigma_prime_eval(c):
        # Use the dialog representation to evaluate sigma and sigma_prime directly.
        eval_w_x, eval_sigma_x = gf(0), gf(1)
        eval_w_x_prime, eval_sigma_x_prime = gf(0), gf(0)
        for coeff, is_swap in dialog:
            if is_swap:
                eval_w_x, eval_sigma_x = eval_sigma_x, eval_w_x
                eval_w_x_prime, eval_sigma_x_prime = eval_sigma_x_prime, eval_w_x_prime
            eval_w_x_prime_times_coeff = eval_w_x_prime * coeff
            eval_sigma_x_prime -= eval_w_x_prime_times_coeff
            eval_sigma_x_prime = eval_sigma_x_prime * c
            eval_sigma_x_prime += eval_sigma_x
            eval_w_x_times_coeff = eval_w_x * coeff
            eval_sigma_x_prime -= eval_w_x_times_coeff
            eval_sigma_x -= eval_w_x_times_coeff
            eval_sigma_x = eval_sigma_x * c
        ct = n // 2 + delta // 2
        eval_sigma_x = eval_sigma_x // c
        eval_sigma_x_prime -= ct * eval_sigma_x
        return eval_sigma_x, eval_sigma_x_prime

    block_length = gf.order - 1
    gammas = gf([gf.primitive_element**i for i in range(block_length)])
    errors = gf.Zeros(block_length)
    for i in range(block_length):
        sigma_x_eval, sigma_x_prime_eval = sigma_and_sigma_prime_eval(gammas[i])
        if sigma_x_eval == 0:
            j = block_length - i if i else i
            const_to_mul = gammas[j] * (gammas[i] ** n)
            errors[j] = const_to_mul * omega_eval(gammas[j]) * sigma_x_prime_eval**-1
    return errors


def rs_syndrome_decoder_eea_divstep(s: np.ndarray) -> np.ndarray:
    gf = type(s[0])
    # Step-1: Compute sigma_x and omega_x using EEA
    x, zero, one = Poly.Identity(gf), Poly.Zero(gf), Poly.One(gf)
    n = len(s)

    # Compute the dialog representation using Bernstein-Yang style GCD iterations.
    u_x, omega_x = one, Poly(s, field=gf)
    dialog = []
    for i in range(n - 1):
        assert u_x.degree + omega_x.degree <= 2 * n - i - 1
        assert omega_x.degree <= n - i // 2 and u_x.degree <= n
        coeff = omega_x(0) // u_x(0)
        if get_delta(dialog) >= 0 and coeff != 0:
            omega_x, u_x = u_x, omega_x
            coeff = coeff**-1
        assert (
            u_x.degree + omega_x.degree <= 2 * n - i
        ), f'{get_delta(dialog)=}, {u_x.degree=}, {omega_x.degree=}, {coeff=}'
        omega_x = omega_x - u_x * coeff
        dialog.append(coeff)
        assert omega_x(0) == 0
        omega_x //= x

    # Use the dialog representation to evaluate sigma and sigma_prime directly.
    def sigma_and_sigma_prime_eval(c):
        eval_w_x, eval_sigma_x = gf(0), gf(1)
        eval_w_x_prime, eval_sigma_x_prime = gf(0), gf(0)
        for i, coeff in enumerate(dialog):
            is_swap = get_delta(dialog[:i]) >= 0 and coeff != 0
            if is_swap:
                eval_w_x, eval_sigma_x = eval_sigma_x, eval_w_x
                eval_w_x_prime, eval_sigma_x_prime = eval_sigma_x_prime, eval_w_x_prime
            eval_sigma_x_prime = (eval_sigma_x - eval_w_x * coeff) + c * (
                eval_sigma_x_prime - eval_w_x_prime * coeff
            )
            eval_sigma_x = (eval_sigma_x - eval_w_x * coeff) * c
        ct = n // 2 + get_delta(dialog) // 2 - 1
        return eval_sigma_x // c**ct, eval_sigma_x_prime // c**ct - ct * eval_sigma_x // c ** (
            ct + 1
        )

    delta = get_delta(dialog)
    block_length = gf.order - 1
    gammas = gf([gf.primitive_element**i for i in range(block_length)])
    errors = gf.Zeros(block_length)
    for i in range(block_length):
        sigma_x_eval, sigma_x_prime_eval = sigma_and_sigma_prime_eval(gammas[i])
        if sigma_x_eval == 0:
            j = block_length - i if i else i
            errors[j] = gammas[j] * (gammas[i] ** (n // 2 - delta // 2 + 1))
            errors[j] = errors[j] * omega_x(gammas[j]) / sigma_x_prime_eval
    return errors


def polynomial_eval_using_fft(p: Poly | galois.FieldArray, n: int):
    if isinstance(p, Poly):
        gf_type = p.field
        q = gf_type.Zeros(n)
        q[: p.degree + 1] = p.coeffs[::-1]
    else:
        gf_type = type(p[0])
        q = gf_type.Zeros(n)
        q[: len(p)] = p
    return np.fft.fft(q)


def chien_search(p: Poly, gammas: galois.Array):
    r"""Chien Search algorithm to find roots of the polynomial p(x) using FFT"""
    q = polynomial_eval_using_fft(p, len(gammas))
    ret = (q == 0).astype(int)
    return ret


def forneys_algorithm(
    omega: Poly, sigma: Poly | galois.FieldArray, sigma_roots: np.ndarray, gammas: galois.Array
):
    r"""Forneys algorithm to compute the error values given error locator polynomial Omega"""
    gf_type = omega.field
    if not isinstance(sigma, Poly):
        sigma = Poly(sigma[::-1], field=type(sigma[0]))
    sigma_prime = sigma.derivative()
    omega_evals = polynomial_eval_using_fft(omega, len(gammas))
    sigma_prime_evals = polynomial_eval_using_fft(sigma_prime, len(gammas))
    errors = gf_type.Zeros(len(sigma_roots))
    for i in range(len(sigma_roots)):
        if sigma_roots[i]:
            j = len(gammas) - i if i else i
            errors[j] = gammas[j] * omega_evals[i] / sigma_prime_evals[i]
    return errors


def rs_syndrome_decoder_eea_divstep_expanded(s: np.ndarray) -> np.ndarray:
    gf, n = type(s[0]), len(s)

    # Step-1: Compute sigma_x and omega_x using EEA on S and x^(2t)
    delta, x = 1, Poly.Identity(gf)
    u_x, omega_x = Poly.One(gf), Poly(s, field=gf)
    w_x, sigma_x = Poly.Zero(gf), Poly.One(gf)
    n_u_x, n_omega_x = len(s) + 1, len(s)

    for i in range(n - 1):
        coeff = omega_x(0) // u_x(0)
        assert delta == n_u_x - n_omega_x
        if delta > 0 and coeff != 0:
            omega_x, u_x = u_x, omega_x
            sigma_x, w_x = w_x, sigma_x
            n_u_x, n_omega_x = n_omega_x, n_u_x
            delta = -delta
            coeff = coeff**-1
        assert n_u_x <= n_omega_x or coeff == 0, f'{n_u_x=}, {n_omega_x=}'
        omega_x = (omega_x - u_x * coeff) // x
        sigma_x = (sigma_x - w_x * coeff) * x
        n_omega_x -= 1
        delta += 1

    n_errors = n // 2 - delta // 2
    omega_x = (omega_x // x).reverse() * x ** (n_errors - omega_x.degree)
    sigma_x //= x ** (n - 1 - n_errors)

    # Step-2: Use sigma and omega polynomials to determine errors.
    gammas = gf([gf.primitive_element**i for i in range(gf.order - 1)])
    sigma_roots = chien_search(sigma_x, gammas)
    errors = forneys_algorithm(omega_x * x, sigma_x, sigma_roots, gammas)

    return errors


def rs_syndrome_decoder_eea_basic(s: np.ndarray) -> np.ndarray:
    gf, n = type(s[0]), len(s)
    # Step-1: Compute sigma_x and omega_x using EEA on S and x^(2t)
    # 2|S| + O(log(|S|))
    x, zero, one = Poly.Identity(gf), Poly.Zero(gf), Poly.One(gf)
    delta, u_x, omega_x, w_x, sigma_x = 0, x ** len(s), Poly(s[::-1], field=gf), zero, one
    for i in range(0, len(s)):
        omega_x, sigma_x, delta = omega_x * x, sigma_x * x, delta - 1
        omega_x_2t = omega_x.coefficients(order='asc', size=len(s) + 1)[len(s)]
        if omega_x_2t != 0 and delta < 0:
            delta, omega_x, u_x, sigma_x, w_x = -delta, u_x, omega_x, w_x, sigma_x
        omega_x_2t = omega_x.coefficients(order='asc', size=len(s) + 1)[len(s)]
        omega_x = omega_x - u_x * (omega_x_2t // u_x.coeffs[0])
        sigma_x = sigma_x - w_x * (omega_x_2t // u_x.coeffs[0])

    # Step-2: Use sigma and omega polynomials to determine errors.
    gammas = gf([gf.primitive_element**i for i in range(gf.order - 1)])
    sigma_roots = chien_search(sigma_x, gammas)
    errors = forneys_algorithm(omega_x * x, sigma_x, sigma_roots, gammas)

    return errors
