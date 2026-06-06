#!/usr/bin/env python3

import numpy as np
from bp import load_instance_from_tsv


def add_high_degree_clauses(instance, degree, num):
  for i in range(num):
    variables = np.random.choice(np.arange(instance['n']), size=degree, replace=False)
    instance['constraints'].append({
      'coefficients': [1]*len(variables),
      'variables': list(variables),
      'sum_values': {np.random.randint(0, 2)},
    })
    instance['m'] += 1

  print(f'Added {num} new clauses of degree {degree}')
  k_eff = np.mean([len(cons["variables"]) for cons in instance["constraints"]])
  print(f'average clause degree k_eff = {k_eff}')
  D_eff = k_eff * instance['m'] / instance['n']
  print(f'average variable degree D_eff = {D_eff}')


def dump_old_format(instance, fname):
  m, n = instance['m'], instance['n']
  with open(fname, 'w') as outfile:
    outfile.write(f'{n}\t{m}\n')
    for i in range(m):
      constraint = instance['constraints'][i]
      v = list(constraint['sum_values'])[0]
      if v:
        outfile.write('-')
      outfile.write('0.5000')
      for j in constraint['variables']:
        outfile.write(f'\t{j}')
      outfile.write('\n')


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser('Add some high-degree clauses to a tsv problem instance')
  parser.add_argument('--in', dest='input_fname', help='problem file in tsv format', required=True, type=str)
  parser.add_argument('--out', help='problem file in tsv format', required=True, type=str)
  parser.add_argument('--degree', help='degree of the clauses to add', type=int, required=True)
  parser.add_argument('--num', help='number of high degree clauses to add', type=int, required=True)
  args = parser.parse_args()
  instance = load_instance_from_tsv(args.input_fname)
  add_high_degree_clauses(instance, args.degree, args.num)
  dump_old_format(instance, args.out)
