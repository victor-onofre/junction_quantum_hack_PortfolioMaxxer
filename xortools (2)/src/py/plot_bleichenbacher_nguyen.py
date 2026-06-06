import numpy as np
import scipy
import scipy.optimize
from scipy.stats import binomtest
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import os
import math
import glob
import gzip
import os
from aggregate_bleichenbacher_nguyen import load_bndata


def get_rate_and_errorbars(num_errors, num_shots, ci=0.95):
    if type(num_errors) is np.ndarray or type(num_shots) is np.ndarray:
        if type(num_errors) is not np.ndarray:
            num_errors = num_errors * np.ones(num_shots.shape, dtype=np.int64)
        if type(num_shots) is not np.ndarray:
            num_shots = num_shots * np.ones(num_errors.shape, dtype=np.int64)
        ps, (los, his) = [], ([], [])
        for ne, ns in zip(list(num_errors), list(num_shots)):
            p, (l, h) = get_rate_and_errorbars(ne, ns, ci)
            ps.append(p)
            los.append(l)
            his.append(h)
        return np.array(ps), (np.array(los), np.array(his))
    assert num_errors <= num_shots
    if num_shots == 0:
        return (0.5, (0.5, 0.5))
    p = num_errors / num_shots
    result = binomtest(n=num_shots, k=num_errors, p=p)
    interval = result.proportion_ci(confidence_level=ci, method="exact")
    return p, (p - interval.low, interval.high - p)


def error_bar(x, y, yerr, color=None, **kwargs):
    plt.errorbar(x, y, yerr=yerr, color=color, **kwargs)


def error_bar_and_fill_between(x, y, yerr, color=None, scatter=False, **kwargs):
    plt.errorbar(x, y, yerr=yerr, color=color, **kwargs)
    plt.fill_between(x, y - yerr[0][:], y + yerr[1][:], color=color, alpha=0.4)
    if scatter:
        plt.scatter(x, y, color=color)


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

red = np.array([196, 30, 58]) / 255
yellow = np.array([255, 191, 0]) / 255
blue = np.array([64, 143, 239]) / 255
green = np.array([50, 205, 50]) / 255
purple = np.array([93, 63, 211]) / 255
orange = np.array([255, 165, 0]) / 255
pink = np.array([255, 119, 255]) / 255
aqua = np.array([93, 218, 161]) / 255
chocolate = np.array([210, 105, 30]) / 255
colors = [chocolate, red, yellow, blue, green, purple, orange, pink, aqua]
# COLORBLIND-FRIENDLY
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
colors = 2 * colorblind_color_cycle


def make_success_probability_plot(datas, prefixes, apply_same_sum=[False]):
    fig = plt.figure(figsize=(5, 4))
    for bool_apply_same_sum in apply_same_sum:
        for data, prefix, c in zip(datas, prefixes, colors):
            p_vals = sorted(set(data["p"]))
            num_shots, num_success = [], []
            for p in p_vals:
                idx = data["apply_same_sum"] == bool_apply_same_sum
                idx &= data["p"] == p
                idx &= data["m"] == p - 1
                idx &= data["n"] == (p - 1) // 2
                idx &= data["num_distractors"] == (p - 3) // 2
                nshot = data["num_shots"][idx].sum()
                idx &= data["max_num_sat"] == data["m"]
                nsucc = data["num_shots"][idx].sum()
                if not nsucc:
                    break
                num_shots.append(nshot)
                num_success.append(nsucc)
            num_shots = np.array(num_shots)
            num_success = np.array(num_success)
            rate, errs = get_rate_and_errorbars(num_success, num_shots)
            sstr = " (no same-sum)" if not bool_apply_same_sum else ""
            plt.errorbar(
                x=p_vals[: rate.shape[0]],
                y=rate,
                yerr=errs,
                color=c,
                linestyle=("-" if bool_apply_same_sum else "--"),
                label=f"{prefix.upper()}{sstr}",
            )
    plt.yscale("log")
    plt.legend()
    plt.title(
        r"Probability of satisfying all constraints versus $p$"
        "\n"
        r"$m=p-1, n=m/2, n_{\mathrm{distractors}}=(p-3)/2$"
    )
    # \delta = 0.99
    plt.ylabel(r"$\mathbb{P}\left[\mathrm{success}\right]$")
    plt.xlabel(r"$p$")
    plt.tight_layout()
    ssenc = "".join(
        str(int(bool_apply_same_sum)) for bool_apply_same_sum in apply_same_sum
    )
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/bleichenbacher_nguyen_success_rate_vs_p_same_sum{ssenc}.jpg",
        bbox_inches="tight",
        dpi=500,
    )
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/bleichenbacher_nguyen_success_rate_vs_p_same_sum{ssenc}.pdf",
        bbox_inches="tight",
    )


def make_plots(data, prefix):
    fig = plt.figure(figsize=(6, 4))
    rate, errs = get_rate_and_errorbars(data["num_success"], data["num_shots"])
    p_vals = sorted(p for p in set(data["p"]) if p != 2)
    colors = matplotlib.cm.cividis(np.linspace(0, 1, len(p_vals)))
    # colors = matplotlib.cm.viridis(np.linspace(0, 1, len(p_vals)))
    for i, p in enumerate(p_vals):
        assert p % 2 == 1
        idx = data["p"] == p
        plt.errorbar(
            x=data["num_distractors"][idx] / p,
            y=rate[idx],
            yerr=(errs[0][idx], errs[1][idx]),
            color=colors[i],
            label=f"$p={p}$",
        )
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.yscale('log')
    plt.title(
        r"Bleichenbacher-Nguyen via "
        f"{prefix.upper()}"
        r": $\delta = 0.99, m=p-1, n=m/2$"
    )
    plt.ylabel(r"$\mathbb{P}\left[\mathrm{success}\right]$")
    plt.xlabel(r"$n_{\mathrm{distractors}} / p$")
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/{prefix}_vs_num_distractors.jpg",
        bbox_inches="tight",
        dpi=500,
    )
    plt.yscale("log")
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/{prefix}_vs_num_distractors_logy.jpg",
        bbox_inches="tight",
        dpi=500,
    )
    plt.clf()
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    idx = (data["num_distractors"] == (data["p"] - 3) // 2) & (data["num_success"] > 0)
    plt.errorbar(
        x=data["p"][idx],
        y=rate[idx],
        yerr=(errs[0][idx], errs[1][idx]),
        color=blue,
        marker="s",
        markerfacecolor="none",
    )
    plt.title(
        r"Probability of satisfying all constraints versus $p$ for Bleichenbacher-Nguyen via "
        f"{prefix.upper()}"
        r":"
        "\n"
        r"$\delta = 0.99, m=p-1, n=m/2, n_{\mathrm{distractors}}=(p-3)/2$"
    )
    plt.ylabel(r"$\mathbb{P}\left[\mathrm{success}\right]$")
    plt.xlabel(r"$p$")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/{prefix}_vs_p.jpg",
        bbox_inches="tight",
        dpi=500,
    )
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/{prefix}_vs_p.pdf",
        bbox_inches="tight",
    )
    plt.clf()
    plt.close(fig)


def make_hist_plots(data, prefix):
    fig = plt.figure(figsize=(6, 4))
    data["frac_sat"] = data["max_num_sat"] / data["m"]
    p_vals = [p for p in sorted(set(data["p"])) if p > 2]
    idx = data["num_distractors"] == (data["p"] - 3) // 2
    frac_sat_dists = [
        2 * (data["frac_sat"][(data["p"] == p) & idx] - 1 / 2) for p in p_vals
    ]
    plt.violinplot(
        dataset=frac_sat_dists, positions=p_vals, vert=True, widths=1.75, showmeans=True
    )
    plt.title(
        r"Normalized objective function value versus $p$ for Bleichenbacher-Nguyen via "
        f"{prefix.upper()}"
        r":"
        "\n"
        r"$\delta = 0.99, m=p-1, n=m/2, n_{\mathrm{distractors}}=(p-3)/2$"
    )
    plt.xlabel(r"$p$")
    plt.ylabel(r"$f(x)/m$")
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/{prefix}_hist_vs_p.jpg",
        bbox_inches="tight",
        dpi=500,
    )
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/{prefix}_hist_vs_p.pdf",
        bbox_inches="tight",
    )
    plt.clf()
    plt.close(fig)


def make_violin_plots(data, prefix, apply_same_sum=True):
    fig = plt.figure(figsize=(6, 4))
    data["frac_sat"] = data["max_num_sat"] / data["m"]
    p_vals = np.array([p for p in sorted(set(data["p"])) if p > 2])
    idx = data["num_distractors"] == (data["p"] - 3) // 2
    idx &= data["apply_same_sum"] == apply_same_sum
    idx &= data["m"] == (data["p"] - 1)
    idx &= data["n"] == data["m"] // 2
    assert (data["m"][idx] == data["p"][idx] - 1).all()
    assert (data["n"][idx] == data["m"][idx] // 2).all()

    # Prepare the data for seaborn violin plot
    plot_data = pd.DataFrame(
        {"p": data["p"][idx], "frac_obj": 2 * (data["frac_sat"][idx] - 1 / 2)}
    )

    sns.violinplot(
        x="p", y="frac_obj", data=plot_data, cut=0, native_scale=True, color=colors[3]
    )

    m_vals = p_vals - 1
    n_vals = m_vals // 2
    num_distractor_vals = (p_vals - 3) // 2
    r_vals = 1 + num_distractor_vals
    th_num_sat = n_vals + (m_vals - n_vals) * (r_vals / p_vals)
    th_frac_obj = 2 * th_num_sat / m_vals - 1

    sns.lineplot(
        x=p_vals,
        y=th_frac_obj,
        color=colors[1],
        marker="o",
        label="Truncation Heuristic",
    )
    plt.title(
        r"Normalized objective function value versus $p$ for Bleichenbacher-Nguyen via "
        f"{prefix.upper()}"
        r":"
        "\n"
        r"$\delta = 0.99, m=p-1, n=m/2, n_{\mathrm{distractors}}=(p-3)/2$"
    )
    plt.xlabel(r"$p$")
    plt.ylabel(r"$f(x)/m$")
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/{prefix}_violin_vs_p_same_sum{int(apply_same_sum)}.jpg",
        bbox_inches="tight",
        dpi=500,
    )
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/{prefix}_violin_vs_p_same_sum{int(apply_same_sum)}.pdf",
        bbox_inches="tight",
    )
    plt.clf()
    plt.close(fig)


def make_m_ndis_plots(data, prefix, apply_same_sum=False):
    fig = plt.figure(figsize=(4, 3))
    x = []
    y = []
    z = []
    for p in [7, 17, 29]:
        if p == 29 and prefix != "lll":
            continue
        assert p % 2 == 1
        idx1 = data["apply_same_sum"] == apply_same_sum
        idx1 &= data["p"] == p
        rel_m = data["m"][idx1] / p
        rel_m_vals = sorted(set(rel_m))
        rel_ndis_vals = sorted(set(data["num_distractors"][idx1] / p))
        # print(f'prefix = {prefix} p = {p} idx1.any() = {idx1.any()} rel_m_vals = {rel_m_vals} rel_ndis_vals = {rel_ndis_vals}')
        X, Y = np.meshgrid(rel_m_vals, rel_ndis_vals)
        Z = np.zeros(shape=X.shape)
        for i, rm in enumerate(rel_m_vals):
            for j, rndis in enumerate(rel_ndis_vals):
                Z[j, i] = np.sum(
                    data["num_shots"][
                        idx1
                        & (data["m"] / p == rm)
                        & (data["num_distractors"] / p == rndis)
                        & (data["min_l22"] >= int(rm * p))
                    ]
                ) / np.sum(
                    data["num_shots"][
                        idx1
                        & (data["m"] / p == rm)
                        & (data["num_distractors"] / p == rndis)
                    ]
                )
        # plt.contour(X, Y, Z)
        plt.imshow(
            Z,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            #  aspect='auto'
        )
        plt.colorbar(label=r"$\mathbb{P}\left[||\mathbf{y}||_2 \geq \sqrt{m}\right]$")
        plt.xlabel(r"$n_{\mathrm{distractors}} / p$")
        plt.ylabel(r"$m/p$")
        plt.title(
            # r'Probability that shortest vector $\mathbf{y}$ has $||\mathbf{y}||_2 = \sqrt{m}$ '
            f"$p = {p}$"
        )
        plt.savefig(
            f"bleichenbacher_nguyen_data/plots/{prefix}_l2_vs_num_distractors_vs_m_p{p}_same_sum_{int(apply_same_sum)}.jpg",
            bbox_inches="tight",
            dpi=500,
        )
        plt.savefig(
            f"bleichenbacher_nguyen_data/plots/{prefix}_l2_vs_num_distractors_vs_m_p{p}_same_sum_{int(apply_same_sum)}.pdf",
            bbox_inches="tight",
            dpi=500,
        )
        plt.clf()
        plt.close(fig)


def make_norm_plots(data, prefix, apply_same_sum=False):
    fig = plt.figure(figsize=(6, 4))
    p_vals = np.array([p for p in sorted(set(data["p"])) if p > 2])
    colors = matplotlib.cm.cividis(np.linspace(0, 1, len(p_vals)))
    # colors = matplotlib.cm.viridis(np.linspace(0, 1, len(p_vals)))
    for i, p in enumerate(p_vals):
        assert p % 2 == 1
        idx = data["apply_same_sum"] == apply_same_sum
        idx &= data["p"] == p
        rel_ndis = data["num_distractors"][idx] / p
        m = p - 1
        rel_l2 = np.sqrt(data["min_l22"][idx]) / np.sqrt(m)
        rel_ndis_vals = sorted(set(list(rel_ndis)))
        rel_l2_means = []
        rel_l2_stds = []
        for rel_ndis_val in rel_ndis_vals:
            rel_l2_means.append(rel_l2[rel_ndis == rel_ndis_val].mean())
            rel_l2_stds.append(rel_l2[rel_ndis == rel_ndis_val].std())

        plt.errorbar(
            x=rel_ndis_vals,
            y=rel_l2_means,
            yerr=rel_l2_stds,
            color=colors[i],
            label=f"$p={p}$",
        )
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.yscale('log')
    plt.title(
        r"Bleichenbacher-Nguyen via "
        f"{prefix.upper()}"
        r": $\delta = 0.99, m=p-1, n=m/2$"
    )
    plt.ylabel(r"$\mathbb{E}\left[||\mathbf{y}||_2 / \sqrt{m}\right]$")
    plt.xlabel(r"$n_{\mathrm{distractors}} / p$")
    plt.savefig(
        f"bleichenbacher_nguyen_data/plots/{prefix}_rel_l2_vs_num_distractors_same_sum_{int(apply_same_sum)}.jpg",
        bbox_inches="tight",
        dpi=500,
    )
    plt.clf()
    plt.close(fig)


# make_plots(load_bndata("bleichenbacher_nguyen_data/lll.csv"), "lll")
# make_plots(load_bndata("bleichenbacher_nguyen_data/lll.csv"), "lll")
# make_plots(load_bndata("bleichenbacher_nguyen_data/bkz.csv"), "bkz")
# make_plots(load_bndata("bleichenbacher_nguyen_data/bkz.csv"), "bkz")
lll_hist_data = load_bndata("bleichenbacher_nguyen_data/lll_hist.csv")
bkz_hist_data = load_bndata("bleichenbacher_nguyen_data/bkz_hist.csv")
bkz_bs15_hist_data = load_bndata("bleichenbacher_nguyen_data/bkz-bs15_hist.csv")
# make_norm_plots(lll_hist_data, "lll", apply_same_sum=False)
# make_norm_plots(lll_hist_data, "lll", apply_same_sum=True)
make_m_ndis_plots(lll_hist_data, "lll", apply_same_sum=False)
make_m_ndis_plots(lll_hist_data, "lll", apply_same_sum=True)
make_m_ndis_plots(bkz_hist_data, "bkz", apply_same_sum=False)
make_m_ndis_plots(bkz_hist_data, "bkz", apply_same_sum=True)
make_m_ndis_plots(bkz_bs15_hist_data, "bkz-bs15", apply_same_sum=False)
make_m_ndis_plots(bkz_bs15_hist_data, "bkz-bs15", apply_same_sum=True)
make_success_probability_plot(
    [lll_hist_data, bkz_hist_data, bkz_bs15_hist_data],
    ["lll", "bkz", "bkz-bs15"],
    apply_same_sum=[False],
)
make_success_probability_plot(
    [lll_hist_data, bkz_hist_data, bkz_bs15_hist_data],
    ["lll", "bkz", "bkz-bs15"],
    apply_same_sum=[True],
)
make_success_probability_plot(
    [lll_hist_data, bkz_hist_data, bkz_bs15_hist_data],
    ["lll", "bkz", "bkz-bs15"],
    apply_same_sum=[True, False],
)

for bool_apply_same_sum in [True, False]:
    make_violin_plots(lll_hist_data, "lll", apply_same_sum=bool_apply_same_sum)
    make_violin_plots(bkz_hist_data, "bkz-bs15", apply_same_sum=bool_apply_same_sum)
    make_violin_plots(bkz_bs15_hist_data, "bkz", apply_same_sum=bool_apply_same_sum)
# make_hist_plots(load_bndata("bleichenbacher_nguyen_data/lll_hist.csv"), "lll")
# make_hist_plots(load_bndata("bleichenbacher_nguyen_data/bkz-bs15_hist.csv"), "bkz-bs15")
# make_hist_plots(load_bndata("bleichenbacher_nguyen_data/bkz_hist.csv"), "bkz")
