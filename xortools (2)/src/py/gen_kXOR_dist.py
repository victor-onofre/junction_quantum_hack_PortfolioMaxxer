#!/usr/bin/env python3

import sys

k = int(sys.argv[1])
n_over_m = float(sys.argv[2])

import scipy

dist = scipy.stats.poisson(k / n_over_m)

degfile_str = f"# nodes\n{k}\n1\n# checks\n"
N = 1000 + 10 * int(k / n_over_m)
cutoff = 1e-7
degfile_str += " ".join(str(i) for i in range(1, N) if dist.pmf(i) > cutoff)
degfile_str += "\n"
degfile_str += " ".join(str(dist.pmf(i)) for i in range(1, N) if dist.pmf(i) > cutoff)
degfile_str += "\n"

print(degfile_str)
