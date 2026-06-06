# Implements the decoder in this paper:
# Verification Codes: Simple Low-Density Parity-Check Codes for Large Alphabets
# Michael Luby Michael Mitzenmacher
# https://www.eecs.harvard.edu/~michaelm/NEWWORK/postscripts/verification.pdf

#!/usr/bin/env python3

import bp
import numpy as np
import galois

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
    'Run belief propagation on a problem instance to '
    'decode combinations of --ell errors. ')
  parser.add_argument(
    '--csv', required=True, type=str,
    help='csv file name to load problem instance from')
  parser.add_argument(
    '--ell', required=False, default=1, type=int,
    help='number of constraints to put in the linear combination')
  parser.add_argument(
    '--num-shots', type=int, default=10000,
    help='number of decoding shots to attempt')
  args = parser.parse_args()
  instance = bp.load_instance_from_csv(args.csv)
  H = np.zeros((instance['m'], instance['n']), dtype=int)
  for i, constraint in enumerate(instance['constraints']):
    for c, u in zip(constraint['coefficients'], constraint['variables']):
      H[i,u] = c
  p = instance['p']
  GFp = galois.GF(p)
  rank = np.linalg.matrix_rank(GFp(H))
  print(f'H.shape = {H.shape}')
  num_distinct_rows = len({tuple(list(row)) for row in H})
  print(f'num_distinct_rows = {num_distinct_rows}')
  print(f'Note: H has rank {rank} out of {H.shape[1]}')
  num_errors = bp.sample_and_decode_shots_nodeverify(H, p, args.ell, args.num_shots)
  print({'num_shots': args.num_shots, 'num_errors': num_errors,
         'instance_fname': args.csv})
