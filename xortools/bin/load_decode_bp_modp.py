#!/usr/bin/env python3

import bp
import numpy as np
import galois
import json


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
    'Run belief propagation on a problem instance to find '
    'the maximum ell such that ell errors can be corrected '
    'with error rate < max-error-rate, but ell + 1 errors cannot')
  parser.add_argument(
    '--csv', required=True, type=str,
    help='csv file name to load problem instance from')
  parser.add_argument(
    '--max-error-rate', required=False, default=1.0, type=float,
    help='maximum error rate before stopping'
  )
  parser.add_argument(
    '--num-shots', type=int, default=10000,
    help='number of decoding shots to attempt for each error rate')
  parser.add_argument(
    '--verbose', action='store_true'
  )
  args = parser.parse_args()
  instance = bp.load_instance_from_csv(args.csv)
  H = np.zeros((instance['m'], instance['n']), dtype=int)
  for i, constraint in enumerate(instance['constraints']):
    for c, u in zip(constraint['coefficients'], constraint['variables']):
      H[i,u] = c
  p = instance['p']
  GFp = galois.GF(p)
  rank = np.linalg.matrix_rank(GFp(H))
  if args.verbose:
    print(f'H.shape = {H.shape}')
  num_distinct_rows = len({tuple(list(row)) for row in H})
  if args.verbose:
    print(f'num_distinct_rows = {num_distinct_rows}')
    print(f'Note: H has rank {rank} out of {H.shape[1]}')
  binary_search_results = bp.binary_search(H, p, args.max_error_rate, args.num_shots,
                                       bp.sample_and_decode_shots_bp, verbose=args.verbose)
  binary_search_results['csv'] = args.csv
  print(json.dumps(binary_search_results))
