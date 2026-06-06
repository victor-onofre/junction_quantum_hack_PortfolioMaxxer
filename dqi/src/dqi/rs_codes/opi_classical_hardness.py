import numpy as np
import scipy.optimize


def channel_capacity_QSC_bits(q, p):
    return np.log2(q) + ((1 - p) * np.log2(1 - p) + p * np.log2(p) - p * np.log2(q - 1))


eps = 1e-10


def max_p_flip_assuming_code_and_decoder_achieve_capacity(rate, q):
    func = lambda x: channel_capacity_QSC_bits(q, x) / np.log2(q) - rate
    max_p = scipy.optimize.bisect(func, eps, 1 - eps - 1 / q)
    return max_p


def semicircle(rho, delta):
    return (np.sqrt(rho * (1 - delta)) + np.sqrt(delta * (1 - rho))) ** 2


def tailbound(n, num_heads, p):
    # Returns natural log
    # assert num_heads / n >= p
    # print(f'n = {n} p = {p} n-num_heads = {n - num_heads}')
    lcdf = scipy.stats.binom(n, 1 - p).logcdf(n - num_heads)
    # print(f'lcdf = {lcdf}')
    return lcdf


def dqi_sat_frac_assuming_capacity(r, p, n_over_m):
    rate = 1 - n_over_m
    ell_over_m = max_p_flip_assuming_code_and_decoder_achieve_capacity(rate, p)
    dqival = semicircle(r / p, ell_over_m)
    return dqival


def dqi_sat_frac_assuming_rs(r, p, n_over_m, gs=False):
    rate = 1 - n_over_m
    # ell_over_m = max_p_flip_assuming_code_and_decoder_achieve_capacity(rate, p)
    if gs:
        ell_over_m = 1 - np.sqrt(1 - n_over_m)
    else:
        # Berlekamp-Massey
        ell_over_m = n_over_m / 2
    dqival = semicircle(r / p, ell_over_m)
    return dqival


def get_runtimes(r, p, m, n_over_m_vals):
    runtimes_th = []
    for n_over_m in n_over_m_vals:
        dqival = np.round(dqi_sat_frac_assuming_capacity(r, p, n_over_m) * m) / m
        num_flips = np.round((1 - n_over_m) * m)
        num_heads_needed = np.round((dqival - n_over_m) * m)
        # assert num_heads_needed >= num_flips/2
        log_runtime_th = -tailbound(num_flips, num_heads_needed, r / p)
        runtimes_th.append(np.exp(log_runtime_th))
    return np.array(runtimes_th)


def get_best_runtimes(p, m, n_over_m_vals, rs=False, gs=False, r_vals=None):
    # Choose the best value of r for each point (best r = get the biggest classical runtime)
    runtimes_th = []
    if r_vals is None:
        r_vals = [*range(1, p, max(1, p // 100))]
    for n_over_m in n_over_m_vals:
        max_log_runtime_th = -1
        ropt = -1
        for r in r_vals:
            if rs:
                dqival = np.round(dqi_sat_frac_assuming_rs(r, p, n_over_m, gs=gs) * m) / m
            else:
                # assume we reach capacity
                dqival = np.round(dqi_sat_frac_assuming_capacity(r, p, n_over_m) * m) / m
            num_flips = np.round((1 - n_over_m) * m)
            num_heads_needed = np.round((dqival - n_over_m) * m)
            # assert num_heads_needed >= num_flips/2
            log_runtime_th = -tailbound(num_flips, num_heads_needed, r / p)
            if log_runtime_th > max_log_runtime_th:
                max_log_runtime_th = log_runtime_th
                ropt = r
        runtimes_th.append((ropt / p, np.exp(max_log_runtime_th)))
    return np.array(runtimes_th)
