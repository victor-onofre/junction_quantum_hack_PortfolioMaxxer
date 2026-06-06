#!/usr/bin/env python3

import pymatching
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import scipy


def load_instance_from_tsv(tsv_fname):
  instance = {}
  with open(tsv_fname, 'r') as infile:
    line = next(infile).strip()
    instance['n'], instance['m'] = line.split()
    instance['n'] = int(instance['n'])
    instance['m'] = int(instance['m'])
    instance['G'] = nx.Graph(n=instance['n'])
    for line in infile:
      weight, u, v = line.strip().split()
      u = int(u)
      v = int(v)
      assert not instance['G'].has_edge(u,v), 'redundant edge'
      instance['G'].add_edge(u, v)
    print(instance['m'])
    print(f"{instance['G'].number_of_nodes()} nodes, "
          f"{instance['G'].number_of_edges()} edges")
    assert instance['G'].number_of_edges() == instance['m']
  return instance

def run_experiment(G, ells, num_shots, halt_on_error_rate=None):
  for ei, e in enumerate(G.edges()):
    G.edges[e]['fault_ids'] = ei
    G.edges[e]['error_probability'] = 0.1
  decoder = pymatching.Matching(G)
  results = []
  for ell in ells:
    assert ell >= 1
    # Check if graph has edges
    if G.number_of_edges() == 0 or ell > G.number_of_edges():
      results.append((ell, 0))
      continue

    edges = np.array([[u, v] for (u, v) in G.edges()])
    edge_us = edges[:, 0]
    edge_vs = edges[:, 1]
    num_edges = len(edges)

    num_errors = 0
    sym_diff_counts = {}
    for shot in tqdm(range(num_shots)):
      # Choose a random subset of ell edges
      syndrome = np.zeros(G.number_of_nodes(), dtype=int)
      
      errors = np.random.choice(np.arange(num_edges), size=ell, replace=False)
      np.add.at(syndrome, edge_us[errors], 1)
      np.add.at(syndrome, edge_vs[errors], 1)
      syndrome %= 2
      correction = decoder.decode(syndrome)
      assert correction.shape[0] == num_edges, (num_edges, correction.shape)
      predicted_errors = set(np.where(correction)[0])
      errors = set(errors)

      sym_diff = errors.union(predicted_errors) - errors.intersection(predicted_errors)
      sym_diff_key = tuple(sorted(sym_diff))
      if sym_diff_key not in sym_diff_counts:
        sym_diff_counts[sym_diff_key] = 0
      sym_diff_counts[sym_diff_key] += 1

      if errors != predicted_errors:
        assert len(set(np.where(correction)[0])) <= len(set(errors))
        num_errors += 1

    sym_diff_probs = np.array([float(count) for key, count in sym_diff_counts.items()])
    assert sym_diff_probs.sum() == num_shots
    sym_diff_probs /= num_shots

    shots_needed = 0.9 * num_shots
    counts = []
    for key, count in sorted(sym_diff_counts.items(), key=lambda o:o[1], reverse=True):
      if shots_needed <= 0:
        break
      counts.append(count)
      shots_needed -= count
      # if count > 1:
      #   print(key, count)
    counts = np.array(counts)
    probs = counts / np.sum(counts)
    results.append({'ell': ell,
                    'num_shots': num_shots,
                    'num_errors': num_errors,
                    'cond_entropy': scipy.stats.entropy(probs)})
    print(results[-1])
    if num_errors/num_shots >= halt_on_error_rate:
      break
  return results

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser('decode some binary maxcut DQI instances using Blossom via PyMatchingV2')
  parser.add_argument('--tsv', type=str, required=True, help='tsv instance file')
  parser.add_argument('--max-ell', type=int, default=10, help='maximum number of errors to try adding')
  parser.add_argument('--max-error-rate', type=float, default=0.1, help='maximum error rate before halting the sweep of l')
  parser.add_argument('--num-shots', type=int, default=10000, help='number of decoding shots to attempt')
  args = parser.parse_args()
  instance = load_instance_from_tsv(args.tsv)
  results = run_experiment(instance['G'], np.arange(1, args.max_ell+1), args.num_shots, args.max_error_rate)
  print(results)
