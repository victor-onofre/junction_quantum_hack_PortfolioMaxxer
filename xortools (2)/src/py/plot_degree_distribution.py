import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import glob
import gzip
import os


# Enable LaTeX rendering for text elements
plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use a serif font family
        "font.serif": [
            "Computer Modern Roman"
        ],  # Specify Computer Modern as the serif font
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",  # Use some helpful packages
    }
)

# Customize further if needed (e.g., font size, figure size)
plt.rcParams.update(
    {
        "font.size": 12,  # Set the font size (adjust to your preference)
        "figure.figsize": (6, 4),  # Set the default figure size (adjust as needed)
    }
)


def read_csv(fname):
    with open(fname, "r") as infile:
        reader = csv.reader(infile, quotechar="|")
        keys = next(reader)
        data = {k: [] for k in keys}
        for row in reader:
            for k, v in zip(keys, map(int, row)):
                data[k].append(v)
    data = {k: np.array(list(map(int, v))) for k, v in data.items()}
    return data


def plot_degrees(data, fname, i="i", t="vertex type", v="Variable"):
    fig = plt.figure(figsize=(3, 2))
    plt.bar(
        data["degree"],
        data["num"] / data["num"].sum(),
        width=0.5,
        facecolor="black",
        edgecolor="black",
    )
    plt.yscale("log")
    plt.xlabel(f"{v} degree ${i}$")
    plt.ylabel(f"{t}")
    # plt.xscale('log')
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_degrees(
        read_csv("var_degrees.csv"),
        "var_degrees.pdf",
        v="Variable",
        i="j",
        t=r"$\Delta_j$",
    )
    plot_degrees(
        read_csv("cons_degrees.csv"),
        "cons_degrees.pdf",
        v="Constraint",
        i="i",
        t=r"$\kappa_i$",
    )
