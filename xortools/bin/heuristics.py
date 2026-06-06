#!/usr/bin/env python3
import bp
import numpy as np
import galois
import itertools


def gauss_heur(instance):
  b = []
  rows = []
  for ci in range(instance['m']):
    constraint = instance['constraints'][ci]
    b.append(np.random.choice(constraint['sum_value']))
    row = np.zeros(instance['n'], dtype=int)
    row[constraint['variables']] = constraint['coefficients']
    rows.append(row)
  Fp = galois.GF(instance['p'])
  rows = Fp(np.array(rows))
  b = Fp(np.array(b))
  constraints = np.random.choice(
    np.arange(instance['m']),
    size=instance['n'], replace=False)
  assert len(set(constraints)) == instance['n']
  x = np.linalg.solve(rows[constraints], b[constraints])
  return x, eval_sol(rows, instance, x)

def local_search(instance, x, t=1):
  rows = []
  for ci in range(instance['m']):
    constraint = instance['constraints'][ci]
    row = np.zeros(instance['n'], dtype=int)
    row[constraint['variables']] = constraint['coefficients']
    rows.append(row)
  Fp = galois.GF(instance['p'])
  rows = Fp(np.array(rows))
  value = eval_sol(rows, instance, x)
  while True:
    found_improvement = False
    for r in range(t):
      for combo in itertools.combinations(range(instance['n']), r=t):
        for coeffs in itertools.product(
          *(r*[range(1, instance['p'])])):
          delta = Fp(np.zeros(instance['n'], dtype=int))
          for c, change in zip(combo, coeffs):
            delta[c] = change
          new_value = eval_sol(rows, instance, x + delta)
          if new_value > value:
            x = x + delta
            print(f'found higher value {new_value}')
            value = new_value
            found_improvement = True
            break
        if found_improvement:
          break
      if found_improvement:
        break
    if not found_improvement:
      break
  return x, value

def eval_sol(rows, instance, x):
  y = rows @ x
  num_sat = 0
  for ci in range(instance['m']):
    constraint = instance['constraints'][ci]
    if y[ci] in constraint['sum_value']:
      num_sat += 1
  return num_sat

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
    'Run classical heuristics on a problem instance mod p')
  parser.add_argument(
    '--csv', required=True, type=str,
    help='tsv file name to load problem instance from')
  args = parser.parse_args()
  instance = bp.load_instance_from_csv(args.csv)
  x, val = gauss_heur(instance)
  print(f'initial val {val}')
  x, val = local_search(instance, x, 1)
  print(f't=1 local search val {val}')
