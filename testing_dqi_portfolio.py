import random

import networkx as nx
from sage.all import GF

from max_constraint_sat import MaxConstraintSat
from max_lin_sat import MaxLinSat

from classical_solvers import SimAnnealSolver
from dqi import Dqi
from decoders import BeliefPropagationDecoder

# Section VI-C: Importing the stronger Information Set Decoder (ISD)
from decoders import InformationSetDecoder

# 1. DEFINE DATA
class Asset:
    def __init__(self, name: str, expected_return: int):
        self.name = name
        self.expected_return = expected_return

rng = random.Random(42)
asset_list = [Asset(f"Asset_{i}", rng.randint(5, 20)) for i in range(6)]

# 2. OPTIMIZED MODEL
portfolio = MaxConstraintSat()
x = [portfolio.new_binary_var(f"x_{i}") for i in range(len(asset_list))]

# STRATEGY: Weight Re-balancing (Section V)
# We set the objective terms to weight 1.
portfolio.add_objective(
    sum(x_i * (asset.expected_return // 5) for x_i, asset in zip(x, asset_list)),
    weight=1
)
portfolio.add_objective(-1 * x[0] * x[1], weight=1)

# STRATEGY: Prioritize Feasibility (Section V)
# We increase the budget constraint weight to 3.
# This ensures DQI prioritizes finding a "valid" portfolio (odd number of assets).
portfolio.add_constraint(sum(x) == 1, mod=2, weight=3)

# 3. TRANSFORM WITH STRONG DECODER
print("Transforming with Information Set Decoder (ISD)...")
# Section VI-C: Specialized decoders provide significantly better DQI performance
# by finding better solutions in the "imperfect decoding regime."
max_linsat_inst = portfolio.to_max_linsat(
    default_decoder_constructor=InformationSetDecoder
)

# 4. ESTIMATE WITH HIGHER DEGREE
dqi = Dqi(max_linsat_inst)

# STRATEGY: Increase Polynomial Degree l (Section II-C)
# We increase l from 1 to 2.
# Equation 1: "The larger the value of l, the stronger the separation 
# between good and bad solutions."
print("Estimating DQI performance at degree l=2...")
est_quality = dqi.estimate_solution_quality(
    l=4, 
    n_decoding_samples=50 
)

# 5. COMPARE
sim_anneal = SimAnnealSolver(portfolio)
print("-" * 40)
print(f"Classical Bench (SimAnneal): {sim_anneal.get_solution_quality()}")
print(f"Optimized DQI Estimate (l=2): {est_quality:.2f}")
print("-" * 40)