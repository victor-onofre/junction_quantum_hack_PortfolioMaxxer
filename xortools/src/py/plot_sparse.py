import matplotlib.pyplot as plt
import networkx as nx
from tsv_format import load_instance_from_tsv


def plot_instance(instance, plot_fname):
    m, n = instance["m"], instance["n"]
    X, Y = range(n), range(n, n + m)
    B = nx.Graph()
    B.add_nodes_from(X, bipartite=0)
    B.add_nodes_from(Y, bipartite=1)
    for i, cons in enumerate(instance["constraints"]):
        for j in cons["variables"]:
            B.add_edge(j, i + n)

    # Generate the bipartite layout
    pos = nx.bipartite_layout(B, X)

    # Adjust figure size if necessary
    fig = plt.figure(figsize=(12, 22))

    # Draw the graph with the layout and set edge transparency
    nx.draw(
        B,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=8,
        edge_color="black",
        alpha=0.05,
    )  # Set edge alpha (transparency)

    # Save the plot to a file
    plt.savefig(plot_fname, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("plot an instance")
    parser.add_argument("instance_file", help="instance in tsv format", type=str)
    parser.add_argument(
        "plot_file", help="plot file name (ext should be pdf, jpg, etc.)", type=str
    )
    args = parser.parse_args()
    instance = load_instance_from_tsv(args.instance_file)
    plot_instance(instance, args.plot_file)
