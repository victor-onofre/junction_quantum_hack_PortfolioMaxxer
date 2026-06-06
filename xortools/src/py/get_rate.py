import numpy as np
import scipy.optimize


def channel_capacity_QSC_bits(q, p):
    return np.log2(q) + ((1 - p) * np.log2(1 - p) + p * np.log2(p) - p * np.log2(q - 1))


eps = 1e-20


def max_p_flip_assuming_code_and_decoder_achieve_capacity(rate, q):
    if rate == 1.0:
        return 0.0
    func = lambda x: channel_capacity_QSC_bits(q, x) / np.log2(q) - rate
    max_p = scipy.optimize.bisect(func, eps, 1 - eps - 1 / q)
    return max_p


def H2(p):
    return scipy.stats.entropy([p, 1 - p], base=2)


def nxor(p, n):
    return (1 - (1 - 2 * p) ** n) / 2


def est_saval_reg(D):
    return min(1, 0.5 + np.exp(0.82) * D**-0.44547 / 2)


def est_saval(check_degrees, check_probabilities):
    assert len(check_degrees) == len(check_probabilities)
    total = 0
    for D, p in zip(check_degrees, check_probabilities):
        total += p * est_saval_reg(D)
    return total


def est_adaptive_saval(rate, node_degrees, node_probabilities):
    cum_prob = 0
    cum_sum_degs = 0
    best_val = 0
    for i in range(len(node_degrees)):
        # consider ignoring degrees higher than this
        d = node_degrees[i]
        p = node_probabilities[i]
        cum_sum_degs += d * p
        cum_prob += p
        avg_node_deg = cum_sum_degs / cum_prob
        avg_check_deg = avg_node_deg / (1 - rate) * cum_prob
        trunc_val = cum_prob * est_saval_reg(avg_check_deg) + (1 - cum_prob) / 2.0
        if trunc_val > best_val:
            print(f"new best trunc val: truncate at d={d}, val = {trunc_val}")
        best_val = max(best_val, trunc_val)
    return best_val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "find the rate etc. of some degree distribution pair"
    )
    parser.add_argument("degfile", type=str)
    args = parser.parse_args()
    with open(args.degfile, "r") as infile:
        split_lines = []
        for line in infile:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            split_lines.append(line.split())

    node_degrees = np.array(list(map(int, split_lines[0])))
    node_probabilities = np.array(list(map(float, split_lines[1])))
    check_degrees = np.array(list(map(int, split_lines[2])))
    check_probabilities = np.array(list(map(float, split_lines[3])))

    # Convert to vertex perspective
    node_probabilities = node_probabilities / node_degrees
    node_probabilities /= node_probabilities.sum()
    check_probabilities = check_probabilities / check_degrees
    check_probabilities /= check_probabilities.sum()

    print(f"node_degrees = {list(node_degrees)}")
    print(f"node_probabilities = {list(node_probabilities)}")
    print(f"check_degrees = {list(check_degrees)}")
    print(f"check_probabilities = {list(check_probabilities)}")

    k_eff = np.dot(node_degrees, node_probabilities)
    D_eff = np.dot(check_degrees, check_probabilities)
    rate = 1 - k_eff / D_eff
    delta_max = max_p_flip_assuming_code_and_decoder_achieve_capacity(rate, 2)
    print(
        f"k_eff = {k_eff} D_eff = {D_eff} rate = {rate} 1-rate = {1-rate} shannon_limit = {delta_max}"
    )
    print(
        f"estimated adaptive sa value: {est_adaptive_saval(rate, node_degrees, node_probabilities)}"
    )
    print(f"estimated th value: {1-rate/2}")
    # exit(0)

    def log_ratio(c, k, D, gamma, eta):
        delta = (1 - gamma) * delta_max
        # zeta = np.sqrt(2*delta)
        zeta = np.sqrt(1 - (1 - 2 * delta) ** 2)
        eps = c * (1 - 2 * eta - zeta) / (2 * k)
        p_clause_flipped = sum(
            nxor(eps, d) * p for d, p in zip(node_degrees, node_probabilities)
        )
        p_clause_sat = eta * p_clause_flipped + (1 - eta) * (1 - p_clause_flipped)
        return (
            np.log2(2 * (p_clause_sat - 1 / 2))
            + (k * (H2(eps) - 1) / D + 1 - H2(1 / 2 + zeta / 2)) / (2 * delta)
            - np.log2(zeta)
        )

    c_vals = np.linspace(0, 1, 100)

    def log_ratio_positive(gamma):
        return (log_ratio(c_vals, k_eff, D_eff, gamma, 0) > 0).any()

    if log_ratio_positive(0) > 0:
        gamma_max = scipy.optimize.bisect(
            lambda gamma: int(log_ratio_positive(gamma)) - 1 / 2, 0, 0.9
        )
        print(
            f"max tolerable gamma = {gamma_max} min tolerable error rate = {(1-gamma_max) * delta_max}"
        )
    else:
        print(f"no tolerable gamma")
