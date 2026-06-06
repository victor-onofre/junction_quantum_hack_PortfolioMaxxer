#!/usr/bin/env python3

from bp import load_instance_from_csv
import numpy as np
import copy


def resampled_with_list_size(instance, list_size):
  modified = copy.deepcopy(instance)
  for i in range(modified['m']):
    modified['constraints'][i]['sum_values'] = sorted(
      np.random.choice(np.arange(modified['p']),
      size=list_size, replace=False))
  return modified

def dump_csv(instance, fname):
  with open(fname, 'w') as outfile:
    outfile.write(f"{instance['n']},{instance['m']},{instance['p']}\n")
    for cons in instance['constraints']:
      outfile.write(','.join(map(str, cons['sum_values'])))
      outfile.write(',')
      outfile.write(','.join(
        f'{xi}:{coeff}'
        for xi, coeff in zip(cons['variables'], cons['coefficients'])))
      outfile.write('\n')

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
    'change the list size and resample the '
    'lists for a CSV DQI instance')
  parser.add_argument('--csv', type=str, help='csv input file', required=True)
  parser.add_argument('--list-size', type=int, help='new list size', required=True)
  parser.add_argument('--out', type=str, help='csv output file', required=True)
  args = parser.parse_args()
  instance = load_instance_from_csv(args.csv)
  modified = resampled_with_list_size(instance, args.list_size)
  dump_csv(
    modified,
    args.out,
  )
