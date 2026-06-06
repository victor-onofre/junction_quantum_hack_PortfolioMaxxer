import numpy as np
import scipy
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import os
import math


def simplify_fraction(numerator, denominator):
    if math.gcd(numerator, denominator) == denominator:
        return int(numerator / denominator)
    elif math.gcd(numerator, denominator) == 1:
        return str(numerator) + "/" + str(denominator)
    else:
        top = numerator / math.gcd(numerator, denominator)
        bottom = denominator / math.gcd(numerator, denominator)
        return str(round(top)) + "/" + str(round(bottom))


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


plt.clf()

green = np.array([130, 158, 54]) / 255
orange = np.array([227, 162, 56]) / 255
red = np.array([212, 85, 78]) / 255
blue = np.array((55, 120, 171)) / 255

colors = [green, orange, red, blue]

# COLORBLIND-FRIENDLY:
colorblind_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
colors = colorblind_color_cycle


def linear_model(x, a, b):
    return a * x + b


coeff_vals = []
rates = []
fig = plt.figure(figsize=(7, 5))
plt.grid(which="both")
for k, D, c, D_min_fit in [
    (9, 10, colors[0], 300),
    (1, 2, colors[1], 300),
    (1, 10, colors[2], 300),
    (1, 100, colors[3], 2e4),
]:
    D_vals = []
    thresholds = []
    threshold_errs = []
    rate = 1 - k / D
    rates.append(rate)
    for bsize in range(1001):
        fname = f"kD_density_sweep/k{k*bsize}_D{D*bsize}.txt"
        if not os.path.exists(fname):
            continue
        with open(fname, "rt") as infile:
            for line in infile:
                if not line.startswith("threshold appears to be at least"):
                    continue
                lo, _, hi = line.strip().split()[-3:]
                lo = float(lo)
                hi = float(hi)
                thresholds.append((lo + hi) / 2)
                threshold_errs.append((hi - lo) / 2)
                D_vals.append(bsize * D)
                break
    D_vals = np.array(D_vals)
    k_vals = (1 - rate) * D_vals
    thresholds = np.array(thresholds)
    idx = np.where(D_vals >= D_min_fit)[0][0]
    popt, pcov = scipy.optimize.curve_fit(
        linear_model, np.log(D_vals[idx:]), np.log(thresholds[idx:])
    )
    a, b = popt
    threshold_errs = np.array(threshold_errs)
    plt.errorbar(
        D_vals[k_vals >= 3],
        thresholds[k_vals >= 3],
        yerr=threshold_errs[k_vals >= 3],
        label=f"$k/D = {simplify_fraction(round(k_vals[0]), round(D_vals[0]))}$",
        color=c,
    )
    plt.plot(
        D_vals,
        np.exp(a * np.log(D_vals) + b),
        label=r"best fit: $" f"{np.exp(b):.4f}" r"\,D^{" f"{a:.4f}" r"}$",
        linestyle="--",
        color=c,
    )
    # exponents = []
    # cutoff = 500
    # for i in range(len(D_vals)-cutoff):
    #   popt, pcov = scipy.optimize.curve_fit(linear_model, np.log(D_vals[i:]), np.log(thresholds[i:]))
    #   a, b = popt
    #   exponents.append(a)
    # plt.plot(D_vals[:-cutoff], exponents, label=f'rate {round(rate, 2)}', color=c)
    coeff_vals.append(np.exp(b))

# plt.plot(np.arange(1, 10001), 1/np.arange(1, 10001), label=r'$1/D$', color='black', linestyle='--')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$D$")
plt.xlim(1, 1e5)
plt.ylabel(r"$\ell / m$")
plt.legend()
plt.savefig("kD_density_sweep.jpg", dpi=300, bbox_inches="tight")
plt.savefig("kD_density_sweep.pdf", bbox_inches="tight")
plt.clf()
plt.close(fig)

fig = plt.figure(figsize=(5, 3))
plt.grid(which="both")
plt.plot(rates, coeff_vals)
plt.savefig("coeff_vs_rate.jpg", dpi=300, bbox_inches="tight")
plt.clf()
plt.close(fig)
