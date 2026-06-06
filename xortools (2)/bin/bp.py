#!/usr/bin/env python3

import numpy as np
import networkx as nx
import ldpc
import scipy


def load_instance_from_tsv_noah_format(tsv_fname):
  instance = {}
  with open(tsv_fname, 'r') as infile:
    line = next(infile).strip()
    instance['n'], instance['m'], instance['p'] = line.split()
    instance['n'] = int(instance['n'])
    instance['m'] = int(instance['m'])
    instance['p'] = int(instance['p'])
    instance['constraints'] = []
    for line in infile:
      line = line.strip().split()
      constraint = {
        'coefficients': [],
        'variables': [],
        'sum_value': int(line[0]),
        'type': line[1],
      }
      for c_u in line[2:]:
        c, u = c_u.split('*')
        constraint['variables'].append(int(u))
        constraint['coefficients'].append(int(c))
      instance['constraints'].append(constraint)
    assert len(instance['constraints']) == instance['m']
  return instance

def load_instance_from_csv(csv_fname):
  instance = {}
  with open(csv_fname, 'r') as infile:
    lines = [line.strip() for line in infile if not line.startswith('#')]
  
  # Parsing the first line for n, m, and p
  n, m, p = map(int, lines[0].split(','))
  instance['n'], instance['m'], instance['p'] = n, m, p

  # Initialize the constraints
  instance['constraints'] = []

  # Processing the constraints
  for line in lines[1:]:
    # Splitting the line at ',' to separate RHS value and coefficients
    parts = line.split(',')
    sum_values = []
    for i in range(len(parts)):
      if ':' in parts[i]:
        break
      sum_values.append(int(parts[i]))
    coeffs = []
    variables = []
    for x in parts[i:]:
      index, coeff = x.split(':')
      coeff = int(coeff)
      index = int(index)
      assert coeff != 0, 'all coeffs should be nonzero'
      coeffs.append(coeff)
      variables.append(index)

    constraint = {
      'coefficients': coeffs,
      'variables': variables,
      'sum_values': set(sum_values),
    }

    # Append the constraint to the list
    instance['constraints'].append(constraint)

  # Verify the number of constraints
  assert len(instance['constraints']) == instance['m']

  return instance

def sample_shot(p, m, ell):
  error = np.zeros(m, dtype=int)
  hyperedges = np.random.choice(np.arange(m), size=ell, replace=False)
  error[hyperedges] = np.random.randint(1, p, size=(ell,))
  return error

class BPDecoder:
  def __init__(self, p, epsilon = 1e-10):
    self.epsilon = epsilon
    self.p = p
    self.inv_mod_p_lookup = np.array([
      pow(i, self.p - 2, self.p) for i in range(p)], dtype=int)

    # Create a 2D array used to optimize adding mod p
    idx = [[] for i in range(self.p)]
    for i in range(self.p):
      for j in range(self.p):
        idx[(i+j)%self.p].append([i,j])
    self.add_mod_p_idx_lookup = np.array(idx)

    self.multiply_mod_p_idx_lookup = np.array([
      (np.arange(self.p) * coeff) % self.p for coeff in range(p)])

  def inv_mod_p(self, val):
    assert val != 0
    return self.inv_mod_p_lookup[val]

  def multiply_mod_p(self, dist, coeff):
    out = np.zeros(self.p, dtype=float)
    if coeff == 0:
      out[0] = 1
      return out
    out[self.multiply_mod_p_idx_lookup[coeff]] += dist
    return out
    # return dist[self.multiply_mod_p_idx_lookup[coeff]]

  # def add_mod_p(self, dist1, dist2):
  #   return modp.add_mod_p(dist1, dist2, self.add_mod_p_idx_lookup, self.p)

  def add_mod_p(self, dist1, dist2):
    return np.sum(np.multiply(
      dist1[self.add_mod_p_idx_lookup[:,:,0]],
      dist2[self.add_mod_p_idx_lookup[:,:,1]]), axis=1)

  def get_current_dist(self, node_inbound_messages, u):
    current_dist = np.ones(self.p, dtype=float)
    for _, dist in node_inbound_messages[u]:
      current_dist *= dist
    current_dist = np.maximum(current_dist, self.epsilon)
    current_dist /= np.sum(current_dist)
    return current_dist

  # In BP decoding, the "nodes" correspond to hyperedges / constraints
  # in the CSP, and the "checks" correspond to the variables in the CSP.
  # The parity check matrix H should be of shape (m, n)
  def sample_and_decode_shot(self, H, p, ell):
    error = sample_shot(p, H.shape[0], ell)
    syndrome = (H.T @ error) % p

    p_flip = ell / H.shape[0]
    node_beliefs = np.zeros((H.shape[0], p), dtype=float)
    node_beliefs[:,0] = 1-p_flip
    node_beliefs[:,1:] = p_flip/(p-1)

    node_inbound_messages = [
      # node_id: [(check_from, distribution)]
      [] for u in range(H.shape[0])
    ]
    
    check_inbound_messages = [
      # check_id: (node_from, distribution)
      [] for v in range(H.shape[1])
    ]

    dist = None
    prev_dist = None
    tol = 1e-5
    max_num_iters = ell
    num_iters = 0
    wheres = [np.where(H[:,v])[0] for v in range(H.shape[1])]
    converged = False
    while True:
      # The channel input is treated as an input edge, but without updating it.
      for u in range(H.shape[0]):
        node_inbound_messages[u].append((
            -1, # this should be a check index, but here we use -1
                # to denote it's from the error distribution
            node_beliefs[u]))

      dist = np.array([
              self.get_current_dist(node_inbound_messages, u)
              for u in range(H.shape[0])])
      converged = (prev_dist is not None and (np.abs(dist - prev_dist) < tol).all())
      if num_iters > max_num_iters or converged:
        # We have converged
        break
      prev_dist = dist
      # Propagate on the node side.
      for u in range(H.shape[0]):
        # Node computes outbound messages to each check that involves it
        for v in np.where(H[u])[0]:
          # Look at all inbound messages to this node that don't come from this check
          pointwise_product_dist = np.ones(p, dtype=float)
          for check_from, dist in node_inbound_messages[u]:
            if check_from == v:
              # This message is from this check, so we skip it.
              continue
            pointwise_product_dist *= dist
          pointwise_product_dist = np.maximum(pointwise_product_dist, self.epsilon)
          pointwise_product_dist /= np.sum(pointwise_product_dist)
          check_inbound_messages[v].append((u, pointwise_product_dist))
        # Clear out the inbox of this node
        node_inbound_messages[u] = []

      # Propagate on the check side.
      # Add the outbounds to each node involved in the check
      for v in range(H.shape[1]):
        for u in wheres[v]:
          # Start with a delta distribution on the observed syndrome value.
          u_dist = np.zeros(p, dtype=float)
          u_dist[syndrome[v]] = 1
          # Look at all inbound messages to this check that don't involve this node, and
          # 'fuzzy subtract' their contribution to the syndrome value
          for check_from, dist in check_inbound_messages[v]:
            if check_from == u:
              # This message is from this node, so we skip it.
              continue
            coeff = H[check_from,v]
            u_dist = self.add_mod_p(u_dist, self.multiply_mod_p(dist, p-coeff))
          # Divide out the coefficient of this error node in the check
          assert H[u,v] != 0
          # Cast to int before computing inverse to avoid overflow with numpy int
          # assert ((int(H[u,v])**(p-2))%p) != 0, (H[u, v], p, (int(H[u,v])**(p-2))%p)
          u_dist = self.multiply_mod_p(u_dist, self.inv_mod_p(H[u,v]))
          # Send this message to the node u
          node_inbound_messages[u].append((v, u_dist))
        # Clear out the inbox of this check
        check_inbound_messages[v] = []
      num_iters += 1

    final_distribution = np.array([
      self.get_current_dist(node_inbound_messages, u)
      for u in range(H.shape[0])])
    predicted = final_distribution.argmax(axis=1)
    
    max_probs = final_distribution.max(axis=1)
    num_low_prob_symbols = np.sum(max_probs < 0.9)
    num_symbols = H.shape[0]
    return {
      'error': bool((predicted != error).any()),
      'converged': bool(converged),
      'num_iters': int(num_iters),
      'num_low_prob_symbols': int(num_low_prob_symbols),
      'num_symbols': int(num_symbols),
    }

class BPFilteringDecoder:
  def __init__(self, p, epsilon = 1e-10):
    self.epsilon = epsilon
    self.p = p
    self.inv_mod_p_lookup = np.array([
      pow(i, self.p - 2, self.p) for i in range(p)], dtype=int)

    # Create a 2D array used to optimize adding mod p
    idx = [[] for i in range(self.p)]
    for i in range(self.p):
      for j in range(self.p):
        idx[(i+j)%self.p].append([i,j])
    self.add_mod_p_idx_lookup = np.array(idx)

    self.multiply_mod_p_idx_lookup = np.array([
      (np.arange(self.p) * coeff) % self.p for coeff in range(p)])

  def inv_mod_p(self, val):
    assert val != 0
    return self.inv_mod_p_lookup[val]

  def multiply_mod_p(self, dist, coeff):
    out = np.zeros(self.p, dtype=float)
    if coeff == 0:
      out[0] = 1
      return out
    out[self.multiply_mod_p_idx_lookup[coeff]] += dist
    return out
    # return dist[self.multiply_mod_p_idx_lookup[coeff]]

  # def add_mod_p(self, dist1, dist2):
  #   return modp.add_mod_p(dist1, dist2, self.add_mod_p_idx_lookup, self.p)

  def add_mod_p(self, dist1, dist2):
    return np.sum(np.multiply(
      dist1[self.add_mod_p_idx_lookup[:,:,0]],
      dist2[self.add_mod_p_idx_lookup[:,:,1]]), axis=1)

  def get_current_dist(self, node_inbound_messages, u):
    current_dist = np.ones(self.p, dtype=float)
    for _, dist in node_inbound_messages[u]:
      current_dist *= dist
    current_dist = np.maximum(current_dist, self.epsilon)
    current_dist /= np.sum(current_dist)
    return current_dist

  # In BP decoding, the "nodes" correspond to hyperedges / constraints
  # in the CSP, and the "checks" correspond to the variables in the CSP.
  # The parity check matrix H should be of shape (m, n)
  def sample_and_decode_shot(self, H, p, n_distractors):
    node_beliefs = np.zeros((H.shape[0], p), dtype=float)
    for i in range(node_beliefs.shape[0]):
      # node_beliefs[i] = [0.64554, 0.34448, 0.00997]
      # node_beliefs[i] = [0.73333, 0.13333, 0.13333]
      node_beliefs[i][0] = 1/2
      node_beliefs[i][np.random.choice(np.arange(1, p))] = 1/2
      # j = 0
      # if np.random.random() < 1/3:
      #   j = np.random.choice(np.arange(1, p))
      # node_beliefs[i][j] = 2/3
      # node_beliefs[i][(j+1)%3] = 1/6
      # node_beliefs[i][(j+2)%3] = 1/6
    # node_beliefs[:,0] = 1/(1+n_distractors)
    # for i in range(node_beliefs.shape[0]):
    #   node_beliefs[i, np.random.choice(np.arange(1, p), size=n_distractors, replace=False)] = 1/(1+n_distractors)

    node_inbound_messages = [
      # node_id: [(check_from, distribution)]
      [] for u in range(H.shape[0])
    ]
    
    check_inbound_messages = [
      # check_id: (node_from, distribution)
      [] for v in range(H.shape[1])
    ]

    dist = None
    prev_dist = None
    tol = 1e-5
    max_num_iters = 100
    num_iters = 0
    wheres = [np.where(H[:,v])[0] for v in range(H.shape[1])]
    converged = False
    while True:
      # The channel input is treated as an input edge, but without updating it.
      for u in range(H.shape[0]):
        node_inbound_messages[u].append((
            -1, # this should be a check index, but here we use -1
                # to denote it's from the error distribution
            node_beliefs[u]))

      dist = np.array([
              self.get_current_dist(node_inbound_messages, u)
              for u in range(H.shape[0])])

      converged = (prev_dist is not None and (np.abs(dist - prev_dist) < tol).all())
      if num_iters > max_num_iters or converged:
        # We have converged
        break
      prev_dist = dist
      # Propagate on the node side.
      for u in range(H.shape[0]):
        # Node computes outbound messages to each check that involves it
        for v in np.where(H[u])[0]:
          # Look at all inbound messages to this node that don't come from this check
          pointwise_product_dist = np.ones(p, dtype=float)
          for check_from, dist in node_inbound_messages[u]:
            if check_from == v:
              # This message is from this check, so we skip it.
              continue
            pointwise_product_dist *= dist
          pointwise_product_dist = np.maximum(pointwise_product_dist, self.epsilon)
          pointwise_product_dist /= np.sum(pointwise_product_dist)
          check_inbound_messages[v].append((u, pointwise_product_dist))
        # Clear out the inbox of this node
        node_inbound_messages[u] = []

      # Propagate on the check side.
      # Add the outbounds to each node involved in the check
      for v in range(H.shape[1]):
        for u in wheres[v]:
          # Start with a delta distribution on the expected syndrome value of 0
          u_dist = np.zeros(p, dtype=float)
          u_dist[0] = 1
          # Look at all inbound messages to this check that don't involve this node, and
          # 'fuzzy subtract' their contribution to the syndrome value
          for check_from, dist in check_inbound_messages[v]:
            if check_from == u:
              # This message is from this node, so we skip it.
              continue
            coeff = H[check_from,v]
            u_dist = self.add_mod_p(u_dist, self.multiply_mod_p(dist, p-coeff))
          # Divide out the coefficient of this error node in the check
          assert H[u,v] != 0
          # Cast to int before computing inverse to avoid overflow with numpy int
          # assert ((int(H[u,v])**(p-2))%p) != 0, (H[u, v], p, (int(H[u,v])**(p-2))%p)
          u_dist = self.multiply_mod_p(u_dist, self.inv_mod_p(H[u,v]))
          # Send this message to the node u
          node_inbound_messages[u].append((v, u_dist))
        # Clear out the inbox of this check
        check_inbound_messages[v] = []
      num_iters += 1

    final_distribution = np.array([
      self.get_current_dist(node_inbound_messages, u)
      for u in range(H.shape[0])])
    # print(final_distribution)
    predicted = ((np.random.randn(*final_distribution.shape)*1e-6) + final_distribution).argmax(axis=1)

    max_probs = final_distribution.max(axis=1)
    num_low_prob_symbols = np.sum(max_probs < 0.9)
    num_symbols = H.shape[0]
    return {
      'error': bool((predicted != 0).any()),
      'converged': bool(converged),
      'num_iters': int(num_iters),
      'num_low_prob_symbols': int(num_low_prob_symbols),
      'num_symbols': int(num_symbols),
    }



##### Begin tests
# Test case
decoder = BPDecoder(7)
for i in range(7):
  dist1 = np.zeros(7)
  dist1[i] = 1

  for j in range(7):
    # Test multiplying dist1 by a constant
    dist3 = decoder.multiply_mod_p(dist1, j)
    dist_expected = np.zeros(7)
    dist_expected[(i*j)%7] = 1
    # print(f'testing i = {i} * j = {j}, expected {dist_expected.round(3)} got {dist3.round(3)}')
    assert np.allclose(dist3, dist_expected)

    # Test adding i and j as delta distributions
    dist2 = np.zeros(7)
    dist2[j] = 1
    dist3 = decoder.add_mod_p(dist1, dist2)
    dist_expected = np.zeros(7)
    dist_expected[(i+j)%7] = 1
    # print(f'testing i = {i}, j = {j}, got dist {dist3.round(3)} expected {dist_expected.round(3)}')
    assert np.allclose(dist3, dist_expected, atol=1e-5)

for i in range(1, 7):
  assert ((decoder.inv_mod_p(i) * i) % 7) == 1
#### End tests

class SymbolFlippingDecoder:
  def __init__(self, p):
    self.p = p
    self.inv_mod_p_lookup = np.array([
      pow(i, self.p - 2, self.p) for i in range(p)], dtype=int)

  def inv_mod_p(self, val):
    return self.inv_mod_p_lookup[val]

  def decode_shot(self, H, p, error):
    syndrome = (H.T @ error) % p
    current_sol = np.zeros(H.shape[0], dtype=int)
    num_flips = 0
    while True:
      improvements = {}
      current_syndrome = (H.T @ current_sol) % p
      for i in range(H.shape[0]):
        delta_and_count = {}
        for j in np.where(H[i])[0]:
          # i is the error
          # j is the check
          delta = ((syndrome[j] - current_syndrome[j]) * self.inv_mod_p(H[i,j])) % p
          if delta not in delta_and_count:
            delta_and_count[delta] = 0
          delta_and_count[delta] += 1
        if delta_and_count.get(0, 0) < max(delta_and_count.values()):
          # We can improve the # sat by changing this symbol only
          best_delta = max(delta_and_count, key=lambda delta: delta_and_count[delta])
          new_value = (current_sol[i] + best_delta) % p
          improvements[(i, new_value)] = delta_and_count[delta] - delta_and_count.get(0, 0)
      if not improvements:
        break
      # Use the best improvement
      best_i, best_new_value = max(improvements, key=lambda k: improvements[k])
      current_sol[best_i] = best_new_value
      num_flips += 1
    syndrome_error_symbols = ((current_syndrome - syndrome) != 0).sum()
    return {
      'error': bool((current_sol != error).any()),
      'num_flips': int(num_flips),
      'syndrome_error': bool(syndrome_error_symbols>0),
      'syndrome_error_symbols': int(syndrome_error_symbols),
    }

  def sample_and_decode_shot(self, H, p, ell):
    error = sample_shot(p, H.shape[0], ell)
    return self.decode_shot(H, p, error)

class NodeVerifyDecoder:
  def __init__(self, p, H):
    self.p = p
    self.H = H
    self.wherecache = [set(np.where(H.T[i])[0]) for i in range(H.shape[1])]
    self.inv_mod_p_lookup = np.array([
      pow(i, self.p - 2, self.p) for i in range(p)], dtype=int)

  def inv_mod_p(self, val):
    return self.inv_mod_p_lookup[val]

  def sample_and_decode_shot(self, ell):
    p = self.p
    error = sample_shot(p, self.H.shape[0], ell)
    syndrome = (self.H.T @ error) % p
    current_sol = np.zeros(self.H.shape[0], dtype=int)
    verified_symbols = set()
    unverified_syndromes = set(range(self.H.T.shape[0]))
    while True:
      current_len_verified = len(verified_symbols)
      for i in list(unverified_syndromes):
        assert i in unverified_syndromes
        if syndrome[i] == 0:
          for j in self.wherecache[i]:
            # verify symbol j
            verified_symbols.add(j)
          unverified_syndromes.remove(i)
        else:
          unverified_symbols_this_check = self.wherecache[i] - verified_symbols
          if len(unverified_symbols_this_check) == 0:
            unverified_syndromes.remove(i)
          elif len(unverified_symbols_this_check) == 1:
            j = list(unverified_symbols_this_check)[0]
            current_sol[j] = syndrome[i]
            for other_j in self.wherecache[i]:
              current_sol[j] -= self.H.T[i,other_j] * current_sol[other_j]
            current_sol[j] *= self.inv_mod_p(self.H.T[i, j])
            current_sol[j] %= p
            verified_symbols.add(j)
            unverified_syndromes.remove(i)
      if len(verified_symbols) == current_len_verified:
        # abort
        break
    # If the system is undeterdetermined, then we failed
    return len(current_sol) - len(verified_symbols) > syndrome.shape[0]
    # return (current_sol != error).any()

def sample_and_decode_shots_nodeverify(H, p, ell, num_shots):
  num_errors = 0
  decoder = NodeVerifyDecoder(p, H)
  for shot in range(num_shots):
    error = decoder.sample_and_decode_shot(ell)
    if error:
      print('error')
      num_errors += 1
    else:
      print('success')
  return num_errors  

def sample_and_decode_shots_symbolflipping(H, p, ell, num_shots, verbose=False):
  decoder = SymbolFlippingDecoder(p)
  counters = {'shots': 0, 'ell': ell, 'm': H.shape[0], 'p': p}
  for shot in range(num_shots):
    result = decoder.sample_and_decode_shot(H, p, ell)
    counters['shots'] += 1
    for k in result:
      if k not in counters:
        counters[k] = 0
      counters[k] += result[k]
    if verbose:
      if result['error']:
        print('error')
      else:
        print('success')
  return counters


def sample_and_decode_shots_bp(H, p, ell, num_shots, verbose=False):
  decoder = BPDecoder(p)
  counters = {'shots': 0, 'ell': ell, 'm': H.shape[0], 'p': p}
  for shot in range(num_shots):
    result = decoder.sample_and_decode_shot(H, p, ell)
    counters['shots'] += 1
    for k in result:
      if k not in counters:
        counters[k] = 0
      counters[k] += result[k]
    if verbose:
      if result['error']:
        print('error')
      else:
        print('success')
  return counters

def binary_search(H, p, max_error_rate, num_shots, decode_func, verbose=False):
  m = H.shape[0]
  min_ell = 0
  max_ell = np.ceil((1-1/p) * m)
  search_log = []
  while max_ell - min_ell > 1:
    ell = round((max_ell + min_ell)/2)
    if verbose:
      print(f'current ell = {ell}')
    results = decode_func(H, p, ell, num_shots, verbose=verbose)
    search_log.append(results)
    error_rate = results['error'] / results['shots']
    assert results['shots'] == num_shots
    if error_rate > max_error_rate:
      max_ell = ell
    else:
      min_ell = ell
    if verbose:
      print(f'search_log = {search_log}')
  return {'search_log': search_log, 'max_error_rate': max_error_rate, 'ell_max': min_ell}


# def analyze_errors(error_bitstrings):
#   print(error_bitstrings.shape)
#   print(np.any(error_bitstrings, axis=1))
#   num_bits = error_bitstrings.shape[1]
#   error_rates_by_bit = np.mean(error_bitstrings, axis=0)
#   bits_high_to_low_error_rate = np.argsort(error_rates_by_bit)[::-1]
#   for i in range(num_bits):
#     error_rate = np.mean(np.any(error_bitstrings[:,bits_high_to_low_error_rate[i:]], axis=1))
#     print(f'if we remove the {i} most noisy bits, we get a word error rate of {error_rate}')

def sample_and_decode_shots_ldpc_bp(H, p, ell, num_shots):
  assert p == 2, 'ldpc module only supports p=2'

  G = nx.Graph()
  G.add_nodes_from(np.arange(H.shape[0]), bipartite=0)
  G.add_nodes_from(np.arange(H.shape[0], H.shape[0]+H.shape[1]), bipartite=1)
  G.add_edges_from([(u, v+H.shape[0]) for (u, v) in zip(*np.where(H))])
  print(f'Girth = {nx.girth(G)}')
  p_flip = ell / H.shape[0]
  # tol = 1e-3
  max_num_iters = 10000
  decoder = ldpc.bposd_decoder(H.T, error_rate=p_flip)
  # decoder = ldpc.bp_decoder(
  #   H.T, error_rate=p_flip,
  #   max_iter=max_num_iters, bp_method='ps',
  #   ms_scaling_factor=0.625)
  num_errors = 0
  # error_counts = np.zeros(H.shape[0], dtype=int)
  # error_bitstrings = []
  for shot in range(num_shots):
    error_vec = sample_shot(p, H.shape[0], ell)
    syndrome = (error_vec @ H) % p
    decoding = decoder.decode(syndrome)
    error = (error_vec != decoding).any()
    # error_bitstrings.append(error_vec!=decoding)
    if error:
      print('error')
      # error_counts[error_vec!=decoding] += 1
      num_errors += 1
    else:
      print('success')
  # print(error_counts[np.argsort(error_counts)[::-1]][:10])
  # print(error_counts[np.argsort(error_counts)[::-1]][-10:])
  # error_bitstrings = np.array(error_bitstrings)
  # analyze_errors(error_bitstrings)
  # return {'overall': num_errors, 'mean': np.mean(error_counts), 'median': np.median(error_counts), 'mode': scipy.stats.mode(error_counts)}
  return num_errors

# def bp_heur(instance):
#   p = instance['p']
#   n = instance['n']
#   m = instance['m']
#   node_beliefs = np.zeros((n, p), dtype=float)
#   node_beliefs[:,:] = 1/p

#   node_inbound_messages = [
#     # node_id: [(check_from, distribution)]
#     [] for u in range(n)
#   ]
  
#   check_inbound_messages = [
#     # check_id: (node_from, distribution)
#     [] for v in range(m)
#   ]

#   dist = None
#   prev_dist = None
#   tol = 1e-5
#   max_num_iters = 100
#   num_iters = 0
#   while True:
#     # The channel input is treated as an input edge, but without updating it.
#     for u in range(n):
#       node_inbound_messages[u].append((
#           -1, # this should be a check index, but here we use -1
#               # to denote it's from the prior distribution
#           node_beliefs[u]))

#     dist = np.array([
#             get_current_dist(node_inbound_messages, u, p)
#             for u in range(n)])
#     converged = (prev_dist is not None and (np.abs(dist - prev_dist) < tol).all())
#     if num_iters > max_num_iters or converged:
#       # We have converged
#       break
#     prev_dist = dist
#     # Propagate on the node side.
#     for u in range(n):
#       # Node computes outbound messages to each check that involves it
#       for v, constraint in enumerate(instance['constraints']):
#         if u not in constraint['variables']:
#           continue
#         # Look at all inbound messages to this node that don't come from this check
#         pointwise_product_dist = np.ones(p, dtype=float)
#         for check_from, dist in node_inbound_messages[u]:
#           if check_from == v:
#             # This message is from this check, so we skip it.
#             continue
#           pointwise_product_dist *= dist
#         pointwise_product_dist /= np.sum(pointwise_product_dist)
#         check_inbound_messages[v].append((u, pointwise_product_dist))
#       # Clear out the inbox of this node
#       node_inbound_messages[u] = []

#     # Propagate on the check side.
#     # Add the outbounds to each node involved in the check
#     for v, constraint in enumerate(instance['constraints']):
#       for u in constraint['variables']:
#         # Start with a delta distribution on the desired value.
#         u_dist = np.zeros(p, dtype=float)
#         u_dist[constraint['sum_value']] = 1
#         # Look at all inbound messages to this check that don't involve this node, and 'fuzzy subtract' them
#         for check_from, dist in check_inbound_messages[v]:
#           if check_from == u:
#             # This message is from this node, so we skip it.
#             continue
#           coeff = constraint['coefficients'][constraint['variables'].index(check_from)]
#           u_dist = add_mod_p(u_dist, multiply_mod_p(dist, -coeff, p), p)
#         # Divide out the coefficient of this error node in the check
#         # Cast to int before computing inverse to avoid overflow with numpy int
#         u_coeff = constraint['coefficients'][constraint['variables'].index(u)]
#         assert ((int(u_coeff)**(p-2))%p) != 0, (u_coeff, p, (int(u_coeff)**(p-2))%p)
#         u_dist = multiply_mod_p(u_dist, (int(u_coeff)**(p-2))%p, p)
#         # Send this message to the node u
#         node_inbound_messages[u].append((v, u_dist))
#       # Clear out the inbox of this check
#       check_inbound_messages[v] = []
#     num_iters += 1

#   predicted = np.array([
#     get_current_dist(node_inbound_messages, u, p)
#     for u in range(n)]).argmax(axis=1)

#   num_satisfied = 0
#   for constraint in instance['constraints']:
#     if (sum(
#         c*predicted[u] for c, u in zip(constraint['coefficients'],
#                             constraint['variables'])
#       ) % instance['p']) == constraint['sum_value']:
#       num_satisfied += 1
#   print(predicted)
#   return num_satisfied
