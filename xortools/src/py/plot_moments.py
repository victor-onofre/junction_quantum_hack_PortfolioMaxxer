import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


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


fig = plt.figure(figsize=(10, 6))
plt.grid(which="both")
all_num_clauses = [10, 20, 30, 40, 50]
colors = matplotlib.cm.viridis(np.linspace(0, 1, len(all_num_clauses)))
markers = ["o", "s", "^", "v", "<", ">", "p", "*", "h", "H", "D", "d"]
for i, num_clauses in enumerate(all_num_clauses):
    with open(f"3_6_{num_clauses}.brute", "r") as infile:
        reader = csv.reader(infile)
        keys = next(reader)
        print(keys)
        num_x = []
        num_sat = []
        for line in reader:
            line = dict(zip(keys, map(int, line)))
            num_x.append(line["num_x"])
            num_sat.append(line["num_sat"])
        num_sat = np.array(num_sat)
        f_x = 2 * num_sat - num_clauses
        p_x = np.array(num_x) / np.sum(num_x)
        num_sat = np.array(num_sat)
        binom_num_sat = np.arange(num_clauses + 1)
        binom_f_x = 2 * binom_num_sat - num_clauses
        binom_p_x = scipy.stats.binom.pmf(binom_num_sat, num_clauses, 1 / 2)
        powers = np.arange(0, num_clauses + 1, 2)
        moments = np.dot(p_x, (f_x.reshape(-1, 1) ** powers.reshape(1, -1)))
        binom_moments = np.dot(
            binom_p_x, (binom_f_x.reshape(-1, 1) ** powers.reshape(1, -1))
        )
        print(moments.shape)
        print(binom_moments.shape)
        rel_errs = np.abs(moments - binom_moments) / np.abs(binom_moments)
        plt.plot(powers / num_clauses, rel_errs, color=colors[i], alpha=0.2)
        plt.scatter(
            powers / num_clauses,
            rel_errs,
            label=f"$m={num_clauses}$",
            color=colors[i],
            marker=markers[i],
            alpha=0.8,
        )

plt.plot(
    10 * [0.084],
    np.linspace(0, 2.2, 10),
    color="gray",
    linestyle="-.",
    label="$i/m = 0.084$",
)
plt.plot(
    10 * [2 * 0.084],
    np.linspace(0, 2.2, 10),
    color="black",
    linestyle="--",
    label="$i/m = 2\cdot 0.084$",
)

plt.title(
    "Relative absolute error in the even moments of a (3, 6)-regular ensemble from the binomial distribution"
)
plt.xlabel(f"$i / m$")
plt.ylabel(
    r"$\frac{|\langle f(x)^i\rangle_{\mathrm{actual}}-\langle f(x)^i\rangle_{\mathrm{binomial}}|}"
    r"{|\langle f(x)^i\rangle_{\mathrm{binomial}}|}$"
)
plt.legend()
# plt.yscale('log')
plt.savefig(f"binom_vs_brute_moments.jpg", dpi=400, bbox_inches="tight")
plt.savefig(f"binom_vs_brute_moments.pdf", bbox_inches="tight")
