import matplotlib.pyplot as plt
import numpy as np


def load_degfile(degfile: str):
    with open(degfile, "r") as infile:
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

    return (node_degrees, node_probabilities, check_degrees, check_probabilities)


def dist_to_vertex_perspective(degs, dist):
    vertex_perspective_dist = dist / degs
    vertex_perspective_dist /= vertex_perspective_dist.sum()
    return vertex_perspective_dist


def to_vertex_perspective(node_degs, node_dist, check_degs, check_dist):
    return dist_to_vertex_perspective(node_degs, node_dist), dist_to_vertex_perspective(
        check_degs, check_dist
    )


def get_rate(node_degs, node_dist, check_degs, check_dist):
    npvp, cpvp = to_vertex_perspective(node_degs, node_dist, check_degs, check_dist)
    k_eff = np.dot(node_degs, npvp)
    D_eff = np.dot(check_degs, cpvp)
    rate = 1 - k_eff / D_eff
    return rate


def plot_cdf(node_degs, node_probs, check_degs, check_probs, fname):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Degree")
    plt.ylabel("Cumulative Probability")
    plt.title("Degree Distribution CDF")
    plt.grid(which="both")
    plt.xscale("log")
    plt.ylim((0, 1))
    for i, (deg, prob) in enumerate(zip(node_degs, node_probs)):
        idx = np.argsort(deg)
        deg = deg[idx]
        prob = prob[idx]
        cum_prob = np.cumsum(prob)
        plt.step(deg, cum_prob, where="post", label=f"nodes[{i}]")
    for i, (deg, prob) in enumerate(zip(check_degs, check_probs)):
        idx = np.argsort(deg)
        deg = deg[idx]
        prob = prob[idx]
        cum_prob = np.cumsum(prob)
        plt.step(deg, cum_prob, where="post", label=f"checks[{i}]")
    plt.legend()
    plt.savefig(fname)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "perform various numerical analyses of a degree distribution file"
    )
    parser.add_argument(
        "degfile", type=str, help="path to the degree distribution file", nargs="+"
    )
    parser.add_argument(
        "--cdf-fname",
        type=str,
        default=None,
        help="if present, produces a plot of the node degree distribution to this fname",
    )
    args = parser.parse_args()

    node_degsets = []
    npvps = []
    check_degsets = []
    cpvps = []
    for fname in args.degfile:
        (node_degs, node_dist, check_degs, check_dist) = load_degfile(fname)
        npvp, cpvp = to_vertex_perspective(node_degs, node_dist, check_degs, check_dist)
        node_degsets.append(node_degs)
        npvps.append(npvp)
        check_degsets.append(check_degs)
        cpvps.append(cpvp)
    if args.cdf_fname is not None:
        plot_cdf(
            node_degsets,
            npvps,
            check_degsets,
            cpvps,
            args.cdf_fname,
        )
