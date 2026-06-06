#!/usr/bin/env python3

import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import fractions
import math
import mpmath
import json


def mpmath_matrix_to_numpy(mpmath_mat):
  # Determine the size of the mpmath matrix
  rows, cols = mpmath_mat.rows, mpmath_mat.cols

  # Create a list of lists (rows) for the NumPy array
  numpy_mat = [[float(mpmath_mat[row, col]) for col in range(cols)] for row in range(rows)]

  # Convert the list of lists to a NumPy array
  return np.array(numpy_mat)

def f_tilde_int_exact(t, w, m, eta):
  r = eta/(1+eta)
  total = [mpmath.mpf(0) for _t in t]
  for s in range(w+1):
    for i, _t in enumerate(t):
      total[i] += (
          (-mpmath.sqrt(eta))**(w-s) *
          mpmath.binomial(_t, s) *
          mpmath.binomial(m-_t, w-s))
  return total
  # maxabs = mpmath.max(mpmath.abs(u) for u in total)
  # return [u / maxabs for u in total]

def powerdiff(t, w, m, eta):
  r = eta/(1+eta)
  return [(_t/m - r)**w for _t in t]

POLY_TYPES = {
  'powerdiff': {
    'help': '(f(x) - m/2)^ℓ -- Polynomials in the original DQI manuscript.',
    'fn': powerdiff,
  },
  'symmetrized': {
    'help': 'See Noah\'s generalized DQI notes. Though more annoying to deal with, '
            'these have the benefit that f^(w) only requires decoding combinations of '
            'exactly w errors.',
    'fn': f_tilde_int_exact,
  }
}

def opt_p_accepts(m, target_vals, ell, include_upper, polyfunc,
                  verbose=False, r=1/2., extra_plots=False):
  assert np.all(target_vals[:-1] < target_vals[1:]), 'target_vals array must be sorted in increasing order'
  t_vals = mpmath.arange(m+1)
  # r = eta/(1+eta)
  eta = r/(1-r)
  allowed_w = list(range(ell+1))
  if include_upper:
    allowed_w += list(range(m-ell, m+1))

  if verbose:
    print('evaluating polynomials')
  poly_evals = mpmath.matrix([
      polyfunc(t_vals, w, m, eta)
      for w in allowed_w
  ])
  sqrt_binom = [mpmath.sqrt(mpmath.binomial(m, t)*r**t*(1-r)**(m-t)) for t in t_vals]
  poly_evals_times_sqrt_binom = mpmath.matrix([
    [poly_evals[i,j] * sqrt_binom[j] for j in range(poly_evals.cols)]
    for i in range(poly_evals.rows)])
  for i in range(poly_evals_times_sqrt_binom.rows):
    norm = mpmath.norm(poly_evals_times_sqrt_binom[i,:])
    poly_evals_times_sqrt_binom[i,:] /= norm
  # norms = np.einsum(
  #   'ij,j,ij->i',
  #   poly_evals,
  #   scipy.stats.binom.pmf(t_vals, m, r),
  #   poly_evals
  # )
  # # poly_evals = np.einsum('ij,i->ij', poly_evals, norms**(-.5))
  # U = np.einsum('ij,j->ij', poly_evals, np.sqrt(scipy.stats.binom.pmf(t_vals, m, r))).astype('float64')
  # U = (poly_evals * np.sqrt(scipy.stats.binom.pmf(t_vals, m, r))).astype(np.float64)
  if verbose:
    print('computing QR decomposition')
  Q, R = mpmath.qr(poly_evals_times_sqrt_binom.T, mode='skinny')
  SBI = Q.T @ Q
  if not np.allclose(mpmath_matrix_to_numpy(SBI), np.eye(ell+1)):
    raise ValueError('Q matrix is not numerically orthonormal, maybe try doubling the number of decimal-places')
  Q = Q.T
  R = R.T
  eigs = mpmath.eig(R, left=False, right=False)
  if verbose:
    print(f'eigenvalues of R = {eigs}')
    print('computing R^-1')
  R_inv = R**-1
  if verbose:
    print('preparing orthonormalized state basis')
  poly_evals_times_sqrt_binom = R_inv @ poly_evals_times_sqrt_binom
  poly_evals_times_sqrt_binom_np = mpmath_matrix_to_numpy(poly_evals_times_sqrt_binom)
  if extra_plots:
    # 2D plot of R change-of-basis matrix
    plt.imshow(mpmath_matrix_to_numpy(mpmath.matrix([
      [mpmath.sign(R_inv[i,j]*mpmath.log(abs(R_inv[i,j]))) for j in range(R_inv.cols)]
      for i in range(R_inv.rows)])))
    plt.colorbar()
    plt.savefig('R.jpg', dpi=400)
    plt.clf()

    # 2d image plot of orthornormalized polynomial basis states
    plt.imshow(poly_evals_times_sqrt_binom_np)
    plt.colorbar()
    plt.savefig('orthonormal_poly_basis_image.jpg', dpi=400)
    plt.clf()

    # 1d plots of orthonormalized polynomial basis states
    num_states = poly_evals_times_sqrt_binom_np.shape[0]
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_states))
    # Create a grid of subplots with shared x-axis
    fig, axs = plt.subplots(num_states, 1, figsize=(10, 2 * num_states), sharex=True)
    for i in range(num_states):
      axs[i].plot(t_vals, poly_evals_times_sqrt_binom_np[i], color=colors[i])
      axs[i].set_ylabel(r'$\psi_{' + f'{i}' + '}$')
      axs[i].set_xlim(0, m)
      axs[i].set_ylim(-1/2, 1/2)
      axs[i].grid(which='both')
    # Label the x-axis on the bottom subplot
    axs[-1].set_xlabel('$t$')
    plt.savefig('orthonormal_poly_basis_1d.jpg', dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close('all')

    # Hamiltonian eigenstates plot
    hamiltonian = mpmath.matrix([[0]*poly_evals_times_sqrt_binom.rows]*poly_evals_times_sqrt_binom.rows)
    for i in range(len(t_vals)):
      hamiltonian += t_vals[i] * poly_evals_times_sqrt_binom[:,i:i+1] @ poly_evals_times_sqrt_binom[:,i:i+1].T
    evals, evecs = mpmath.eigh(hamiltonian)
    mpmath.mp.eig_sort(evals, evecs)
    num_states = evecs.cols
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_states))
    fig, axs = plt.subplots(num_states, 1, figsize=(10, 2 * num_states), sharex=True)
    for i in range(num_states):
      evaluated = mpmath_matrix_to_numpy(poly_evals_times_sqrt_binom.T @ evecs[:,i:i+1]).T.reshape(-1)
      axs[i].plot(t_vals, evaluated, color=colors[i],
                  label=r'$\langle  \phi_{'f'{i}'r'}^\dag H \phi_{'f'{i}'r'} \rangle = 'f'{round(evals[i], 3)}''$')
      axs[i].set_ylabel(r'$\phi_{' + f'{i}' + '}$')
      axs[i].grid(which='both')
      axs[i].set_xlim(0, m)
      axs[i].set_ylim(-1/2, 1/2)
      axs[i].legend()
    axs[-1].set_xlabel('$t$')
    plt.savefig('hamiltonian_eigenvectors.jpg', dpi=400, bbox_inches='tight', facecolor='w')
    plt.clf()
    plt.close('all')

  p_accepts = []
  top = mpmath.matrix([[0]*poly_evals_times_sqrt_binom.rows]*poly_evals_times_sqrt_binom.rows)
  # top_np = np.zeros(2*(poly_evals_times_sqrt_binom_np.shape[0],), dtype=poly_evals_times_sqrt_binom_np.dtype)
  prev_target = m
  num_states = len(target_vals)
  colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_states))
  for i, target in enumerate(target_vals[::-1]):
    if verbose:
      print(f'target = {target}')
    top += poly_evals_times_sqrt_binom[:,target:prev_target] @ poly_evals_times_sqrt_binom[:,target:prev_target].T
    # top += np.einsum('ij,j,kj->ik', poly_evals[:,target:prev_target],
    #                  binomial_dist[target:prev_target], poly_evals[:,target:prev_target])
    # top_np += poly_evals_times_sqrt_binom_np[:,target:prev_target] @ poly_evals_times_sqrt_binom_np[:,target:prev_target].T
    prev_target = target
    # assert not np.isnan(top.sum()) and not np.isinf(top.sum())
    # assert not np.isnan(bottom.sum()) and not np.isinf(bottom.sum())
    # evals = scipy.linalg.eigh(top_np, eigvals_only=True)
    evals, evecs = mpmath.eigsy(top)
    p_accept = float(max(evals))
    p_accepts.append(p_accept)
    if extra_plots:
      # 1d plot of optimal polynomial state
      evaluated = mpmath_matrix_to_numpy(
        poly_evals_times_sqrt_binom.T @ evecs[:,evecs.cols-1:]).T.reshape(-1)
      if np.sum(evaluated) < 0:
        evaluated *= -1
      fig = plt.figure(figsize=(10, 5))
      plt.plot(t_vals, evaluated, color=colors[i])
      plt.xlabel(r'$t$')
      plt.ylabel(r'$\psi$')
      plt.grid(which='both')
      plt.xlim(0, m)
      plt.ylim(-1/2, 1/2)
      xt, xtl = plt.xticks()
      xt = np.append(xt, target)
      xtl = np.append(xtl, f'\n$t={target}$')
      plt.xticks(xt, xtl)
      plt.title(f'target: $t = {target}$ acceptance probability: {float(p_accept)}\n'
                f'$m={m}$ $\ell={ell}$')
      plt.savefig(f'optimal_states/{str(len(target_vals)-1-i).zfill(4)}.png',
                  dpi=300, bbox_inches='tight')
      plt.clf()
      plt.close('all')
  return np.array(p_accepts)[::-1]

def plot_opt_curve(target_objective_values, optimal_acceptance_probabilities,
                   m, ell, r, polytype):
  plt.clf()
  plt.plot(
    (target_objective_values/m-r) / np.sqrt(ell/m),
    optimal_acceptance_probabilities,
    color='red')
  plt.xlabel(r'Relative Objective Value: $(t/m-r)/\sqrt{\ell/m}$')
  xt, xtl = plt.xticks()
  xt = np.append(xt, 1/np.sqrt(2))
  xtl = np.append(xtl, '\n'+r'$2^{-1/2}$')
  plt.xticks(xt, xtl)
  plt.ylabel('$p_\mathrm{accept}$')
  plt.title('Optimal Acceptance Probability vs. Target Objective Value\n'
            f'$m={m}$ $\ell={ell}$ $r={r}$')
  plt.grid(which='both')
  plt.savefig('opt_poly_curve.jpg', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
    'optimize the polynomials used in DQI')
  parser.add_argument(
    '--ell', required=True, type=int,
    help='maximum ℓ such that combinations of ℓ errors can be uniquely decoded w.h.p.')
  parser.add_argument(
    '--include-upper', required=False, default=False, action='store_true',
    help='Ordinarily, DQI just uses a linear combination of the low-weight errors, '
    'i.e. combinations of <= ℓ errors. --include-upper says to also allow combinations '
    'of m-ℓ to m errors.'
  )
  parser.add_argument(
    '--polytype', type=str, default='powerdiff', required=False,
    help='Available values:\n' + '\n'.join(
      f'{k}: {v["help"]}' for k, v in POLY_TYPES.items()
    )
  )
  parser.add_argument(
    '--t-min', default=0, type=int,
    help='Computes optimal polynomials for each target t in range(t_min, t_max, t_step).'
  )
  parser.add_argument(
    '--t-max', default=None, type=int,
    help='(default m+1) Computes optimal polynomials for each target t in range(t_min, t_max, t_step).'
  )
  parser.add_argument(
    '--t-step', default=1, type=int,
    help='Computes optimal polynomials for each target t in range(t_min, t_max, t_step).'
  )
  parser.add_argument(
    '--m', required=True, type=int,
    help='number of constraints')
  parser.add_argument(
    '--r', type=float, default=1/2., required=False,
    help='expected value of the clause functions f_i over the problem domain')
  parser.add_argument(
    '--decimal-places', type=int, default=1000,
    help='number of digits of precision (base 10) for calculations'
  )
  parser.add_argument(
    '--extra-plots', default=False, action='store_true',
    help='generate extra plots of the eigenspectra and optimal polynomials'
  )
  args = parser.parse_args()
  mpmath.mp.dps = args.decimal_places
  if args.t_max is None:
    args.t_max = args.m + 1
  target_vals = np.arange(args.t_min, args.t_max, args.t_step)
  p_accepts = opt_p_accepts(
    args.m, target_vals, args.ell, args.include_upper,
    POLY_TYPES[args.polytype]['fn'], r=args.r,
    extra_plots=args.extra_plots,
  )
  if args.extra_plots:
    plot_opt_curve(target_vals, p_accepts, args.m, args.ell, args.r,
                  args.polytype)
  print(json.dumps({
    'target_vals': list(map(int, target_vals)),
    'p_accepts': list(p_accepts),
    'm': args.m,
    'r': args.r,
    'ell': args.ell,
    'decimal_places': args.decimal_places,
    'polytype': args.polytype,
    'include_upper': args.include_upper,
  }))
