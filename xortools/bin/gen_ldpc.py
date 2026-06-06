#!/usr/bin/env python3

import numpy as np
import bp

def parity_check_matrix(n_code, d_v, d_c, seed=None):
  """
  Build a regular Parity-Check Matrix H following Callager's algorithm.

  Parameters
  ----------
  n_code: int, Length of the codewords.
  d_v: int, Number of parity-check equations including a certain bit.
      Must be greater or equal to 2.
  d_c: int, Number of bits in the same parity-check equation. d_c Must be
      greater or equal to d_v and must divide n.
  seed: int, seed of the random generator.

  Returns
  -------
  H: array (n_equations, n_code). LDPC regular matrix H.
      Where n_equations = d_v * n / d_c, the total number of parity-check
      equations.

  """
  seed = 1909923237
  rng = np.random.RandomState(seed)

  if d_v <= 1:
    raise ValueError("""d_v must be at least 2.""")

  if d_c <= d_v:
    raise ValueError("""d_c must be greater than d_v.""")

  if n_code % d_c:
    raise ValueError("""d_c must divide n for a regular LDPC matrix H.""")

  n_equations = (n_code * d_v) // d_c

  block = np.zeros((n_equations // d_v, n_code), dtype=int)
  H = np.empty((n_equations, n_code))
  block_size = n_equations // d_v

  # Filling the first block with consecutive ones in each row of the block

  for i in range(block_size):
    for j in range(i * d_c, (i+1) * d_c):
      block[i, j] = 1
  H[:block_size] = block

  # Create remaining blocks by permutations of the first block's columns:
  for i in range(1, d_v):
    H[i * block_size: (i + 1) * block_size] = rng.permutation(block.T).T
  H = H.astype(int)
  return H

if __name__ == '__main__':
  D = 150
  k = 15
  bsize = 10
  p = 13
  H = parity_check_matrix(bsize*D, k, D)
  for r, c in zip(*np.where(H)):
    assert H[r,c] == 1
    H[r,c] = np.random.randint(1, p)
  print(H)
  print(H.shape)
  print(bp.sample_and_decode_shots_bp(H.T, p, 8, 1000))
