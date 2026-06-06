import tempfile
import scipy.optimize
import subprocess
import os
import numpy as np
from irreg_util import to_vertex_perspective, get_rate

DENSITY_BIN = "./bazel-bin/src/density"


def med_deg(degs, dist):
    idx = np.argsort(degs)
    dist = dist[idx]
    degs = degs[idx]
    cdf = np.cumsum(dist)
    return degs[np.searchsorted(cdf, 0.5, side="right")]


def med_degs(node_degs, node_dist, check_degs, check_dist):
    npvp, cpvp = to_vertex_perspective(node_degs, node_dist, check_degs, check_dist)
    return med_deg(node_degs, npvp), med_deg(check_degs, cpvp)


def write_degfile(node_degs, node_dist, check_degs, check_dist, fp):
    fp.write("# nodes\n")
    fp.write(" ".join(str(d) for d in node_degs) + "\n")
    fp.write(" ".join(str(p) for p in node_dist) + "\n")
    fp.write("# checks\n")
    fp.write(" ".join(str(d) for d in check_degs) + "\n")
    fp.write(" ".join(str(p) for p in check_dist) + "\n")
    fp.flush()


def density_evolution(node_degs, node_dist, check_degs, check_dist):
    with tempfile.NamedTemporaryFile(mode="w+t") as fp:
        write_degfile(node_degs, node_dist, check_degs, check_dist, fp)
        cmd = f"{DENSITY_BIN} --degfile {fp.name} --iterations 50 --bins 301 --shrink-factor 0.1 --precision 0.0001 --max-delta 0.11"
        output = subprocess.check_output(cmd.split()).decode("ascii")
        output = output.split()
        assert output[9] == "threshold"
        assert output[16] == "-"
        return float(output[15])


def fitness(x, node_degs, check_degs, min_rate):
    node_dist = x[: node_degs.shape[0]]
    node_dist /= node_dist.sum()
    check_dist = x[node_degs.shape[0] :]
    check_dist /= check_dist.sum()

    constraint_violation = 0

    # Constrain the rate
    rate = get_rate(node_degs, node_dist, check_degs, check_dist)
    if rate < min_rate:
        constraint_violation += min_rate - rate

    if rate > min_rate * 1.2:
        constraint_violation += rate - min_rate * 1.2

    # # Constraint to force lots of high degree nodes and checks
    # npvp, cpvp = to_vertex_perspective(node_degs, node_dist, check_degs, check_dist)

    # min_deg_high_deg_nodes = 30
    # min_frac_high_deg_nodes = 0.50
    # frac_high_degree_nodes = npvp[node_degs >= min_deg_high_deg_nodes].sum()
    # if frac_high_degree_nodes < min_frac_high_deg_nodes:
    #     constraint_violation += min_frac_high_deg_nodes - frac_high_degree_nodes

    # min_deg_high_deg_checks = 100
    # min_frac_high_deg_checks = 0.90
    # frac_high_degree_checks = cpvp[check_degs >= min_deg_high_deg_checks].sum()
    # if frac_high_degree_checks < min_frac_high_deg_checks:
    #     constraint_violation += min_frac_high_deg_checks - frac_high_degree_checks

    if constraint_violation > 0:
        return constraint_violation

    threshold = density_evolution(node_degs, node_dist, check_degs, check_dist)
    return -threshold


def optimize(node_degs, check_degs, min_rate):
    bounds = [(0, 1) for i in range(node_degs.shape[0] + check_degs.shape[0])]
    args = (node_degs, check_degs, min_rate)

    def callback(intermediate_result=None):
        print(intermediate_result)
        node_dist = intermediate_result.x[: len(node_degs)]
        node_dist /= node_dist.sum()
        check_dist = intermediate_result.x[len(node_degs) :]
        check_dist /= check_dist.sum()
        with open("tmp_intermediate_optimization_result.deg", "w") as outfile:
            write_degfile(node_degs, node_dist, check_degs, check_dist, outfile)

    res = scipy.optimize.differential_evolution(
        fitness, bounds, args=args, workers=42, maxiter=10000, callback=callback
    )
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "design a degree distribution using scipy optimize"
    )
    parser.add_argument("--min-rate", type=float, required=True)
    parser.add_argument("out_degfile", type=str)
    args = parser.parse_args()

    node_degs = set()
    for deg in [20, 21, 25, 26, 30, 31, 40, 41, 45, 46, 48, 49, 50, 51, 3072, 3073]:
        node_degs.add(deg)
    for i in np.linspace(1, 40, 100):
        for k in range(3):
            node_degs.add(int(np.round(2**i)) + k)
    for i in np.linspace(1, 26, 50):
        for k in range(3):
            node_degs.add(int(np.round(3**i)) + k)
    node_degs = list(sorted(node_degs))
    node_degs = np.array(node_degs)
    node_degs = node_degs[node_degs >= 20]
    check_degs = node_degs[node_degs >= 40]
    res = optimize(node_degs, check_degs, 0.5)
    node_dist = res.x[: len(node_degs)]
    node_dist /= node_dist.sum()
    check_dist = res.x[len(node_degs) :]
    check_dist /= check_dist.sum()
    with open(args.out_degfile, "w") as outfile:
        write_degfile(node_degs, node_dist, check_degs, check_dist, outfile)
