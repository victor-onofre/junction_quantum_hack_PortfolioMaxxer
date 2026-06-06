import numpy as np
import stim
import sys
import time
from tqdm.auto import tqdm
from heapdict import heapdict
from tsv_format import load_instance_from_tsv

np.random.seed(1428467)

INF = float('inf')

# # generate a large surface code block with p = 0.002 noise
# p = 0.002
# d = 31
# circuit = stim.Circuit.generated(
#   "surface_code:rotated_memory_x",
#   distance=d,
#   # 2d rounds, suitable for overlapping window decoding
#   rounds=2*d,
#   after_clifford_depolarization=p,
#   before_round_data_depolarization=p,
#   before_measure_flip_probability=p,
#   after_reset_flip_probability=p,
# )
# dem = stim.DetectorErrorModel(
#   gqec.decoding.fix_boundary_edges(
#     circuit.detector_error_model(decompose_errors=True)))

instance_fname = sys.argv[1]
# dem = circuit.detector_error_model(decompose_errors=False)

num_shots = 100

# dets, obs = circuit.compile_detector_sampler(
#   seed=27123839530,
# ).sample(
#     shots=num_shots,
#     separate_observables=True,
#     bit_packed=True,
# )
# dets_unpacked = np.unpackbits(
#     dets, bitorder="little", axis=1, count=dem.num_detectors
# )
instance = load_instance_from_tsv(instance_fname)
errors = []
for cons in instance['constraints']:
  errors.append({
    'detectors': cons['variables'],
    'likelihood_cost': 1.0,
  })
ell = int(sys.argv[2])

sampled_errs = np.zeros((num_shots, instance['m'],), dtype=bool)
dets = np.zeros((num_shots, instance['n'],), dtype=bool)
for shot in range(num_shots):
  sampled_errors = np.random.choice(np.arange(len(errors)), size=ell, replace=False)
  sampled_errs[shot][sampled_errors] = True
  for ei in sampled_errors:
    for detector in errors[ei]['detectors']:
      dets[shot][detector] ^= True

print(np.where(sampled_errs[0])[0])
dets_unpacked = dets.copy()

num_detections_before = np.sum(dets_unpacked)
assert dets.shape[0] == num_shots
print(' '.join(f'D{D}' for D in np.where(dets_unpacked[shot])[0]))
# exit(0)

d2e = [[] for _ in range(dets.shape[1])]
for ei, error in enumerate(errors):
  for d in error['detectors']:
    d2e[d].append(ei)

ecosts = np.array([error['likelihood_cost'] for error in errors])
edets = [error['detectors'] for error in errors]
edets_sets = [set(edets[ei]) for ei in range(len(errors))]
eneighbors = [
  sorted(list({
    od
    for d in edets[ei]
    for oei in d2e[d]
    for od in edets[oei]
    if od not in edets_sets[ei]
  }))
  for ei in range(len(errors))
]
dneighbors = [
  np.array(list({
    od
    for ei in d2e[d]
    for od in edets[ei]
    if od != d
  }))
  for d in range(dets.shape[1])
]

basic_det_costs = [INF for _ in range(dets.shape[1])]
for ei in range(len(edets)):
  for d in edets[ei]:
    basic_det_costs[d] = min(basic_det_costs[d], ecosts[ei] / len(edets[ei]))
  

def decode(dets, det_beam=INF):
  # Min-heap priority queue
  pq = heapdict()

  # Add the START node (empty set of errors)
  errs = np.zeros(len(errors), dtype=bool)

  initial_cost = sum(
    min(
      ecosts[oei] / sum(
        dets[d]
        for d in edets[oei]
      )
      for oei in d2e[d]
    )
    for d in np.where(dets)[0]
  )

  # To each visited node we ascribe an arbitrary index.
  errs_indices = {}
  def get_node_key(errs):
    return tuple(np.where(errs)[0])

  next_unused_node_index = 0
  def get_node_index(node):
    nonlocal next_unused_node_index
    if node in errs_indices:
      return errs_indices[node]
    errs_indices[node] = next_unused_node_index
    next_unused_node_index += 1
    return next_unused_node_index - 1

  def get_detcost(d, errs, dets, det_counts):
    return min([
      # TODO: could use better secondary detcosts in the future
      # (ecosts[oei] + sum(basic_det_costs[od] for od in edets[oei] if not dets[od])) / det_counts[oei]
      ecosts[oei] / det_counts[oei]
      for oei in d2e[d]
      if not errs[oei]
    ] + [INF])

  # The pq stores node entries.
  # The data for each node is (Errors_activated_so_far, Detections_left_over)
  # The Detections_left_over is a pure function of the Errors_activated_so_far.
  # Therefore, those errors can serve as a unique key for the errs_indices.
  # The remaining values are stored in a lookup table, errs_data.

  start_node_key = get_node_key(errs)
  # Should be zero
  start_index = get_node_index(start_node_key)
  assert start_index == 0
  pq[start_index] = (initial_cost, dets.sum())
  blocked_errs = np.zeros(len(errors), dtype=bool)
  det_counts = np.zeros(len(errors), dtype=np.uint8)
  for d in np.where(dets)[0]:
    for oei in d2e[d]:
      det_counts[oei] += 1
  node_data = {start_index: (errs, blocked_errs, dets, det_counts,)}

  # The main outstanding question seems to be whether errs should be stored in a dense or sparse format.

  visited = set()

  num_pq_pushed = 1
  min_num_dets = dets.sum()
  while pq:
    max_num_dets = min_num_dets + det_beam

    # Get the next node to visit
    node_index, (cost, num_dets) = pq.popitem()

    # Mark it as visited
    visited.add(node_index)

    if num_dets > max_num_dets:
      # Since the node graph is a tree, this node is no longer reachable so we can delete its data
      del node_data[node_index]
      continue
    min_num_dets = min(min_num_dets, dets.sum())

    # Lookup the data for this error set node
    errs, blocked_errs, dets, det_counts = node_data[node_index]

    # Since the node graph is a tree, this node is no longer reachable so we can delete its data
    del node_data[node_index]
    # if np.random.random() < 0.01:
    print(f'dets.sum() = {dets.sum()} max_num_dets = {max_num_dets} errs.sum() = {errs.sum()} cost = {cost} len(pq) = {len(pq)} len(node_data) = {len(node_data)}')
    if not dets.any():
      # We have reached an EXIT node
      print(f'Done: cost {cost} num_pq_pushed = {num_pq_pushed}')
      return errs

    # To reduce degree of the graph, we only allow adding errors incident to
    # the lowest index activated detector
    min_det = np.argmax(dets)
    next_blocked_errs = blocked_errs.copy()
    for ei in d2e[min_det]:
      # We limit the number of paths that reach a given node
      # by blocking all errors of a lower index in d2e[min_det]
      # All nodes are still reachable by a correctly ordered choice of the
      # eis to activate.
      next_blocked_errs[ei] = True

      if errs[ei] or blocked_errs[ei]:
        # Skip the already-activated errors or the blocked errors
        continue

      next_errs = errs.copy()
      next_det_counts = det_counts.copy()

      # Activate this error
      next_errs[ei] = True

      # next_dcost, and next_dets.sum() are pure functions are next_errs
      # Therefore, the next_cost is a pure function of next_errs
      next_node_key = get_node_key(next_errs)
      next_node_index = get_node_index(next_node_key)

      # Skip visited nodes
      if next_node_index in visited:
        # TODO: the visited set can be removed entirely, it is just kept for now
        # so that this assertion will be exercised.
        assert False, 'there should be a unique path to each node'
        continue

      # Since the next_cost is a pure function of next_errs, we will never update its priority
      # once it is added to the queue.
      if next_node_index in pq:
        assert False, 'there should be a unique path to each node'
        continue

      next_dets = dets.copy()

      next_cost = cost + ecosts[ei]
      # TODO make it even more incremental by thinking in terms of error overlaps
      # Just apply the det flips
      for d in edets[ei]:
        # Flip this detector
        if next_dets[d]:
          next_dets[d] = False
          for oei in d2e[d]:
            next_det_counts[oei] -= 1
        else:
          next_dets[d] = True
          for oei in d2e[d]:
            next_det_counts[oei] += 1

      assert len(set(edets[ei])) == len(edets[ei])

      if next_dets.sum() > max_num_dets:
        continue

      for d in edets[ei]:
        # Flip this detector
        if dets[d]:
          # We turned off this detector, must subtract the old detcost
          old_detcost_d = get_detcost(d, errs, dets, det_counts)
          next_cost -= old_detcost_d
        else:
          # We turned on this detector, must add the new detcost
          new_detcost_d = get_detcost(d, next_errs, next_dets, next_det_counts)
          next_cost += new_detcost_d

      for od in eneighbors[ei]:
        # Also must update the detcosts of other activated detectors
        # If it was flipped or it is inactive, it will already have been handled in the above loop,
        # or else it is 0 both times in which case it will not contribute to the detcost
        if not dets[od] or not next_dets[od]:
          continue
        assert dets[od] and next_dets[od]
        old_detcost_od = get_detcost(od, errs, dets, det_counts)
        new_detcost_od = get_detcost(od, next_errs, next_dets, next_det_counts)
        next_cost -= old_detcost_od
        next_cost += new_detcost_od

      if next_cost == INF:
        continue
      assert next_cost < INF
      node_data[next_node_index] = (next_errs, next_blocked_errs, next_dets, next_det_counts,)
      num_pq_pushed += 1
      pq[next_node_index] = (next_cost, next_dets.sum(),)
  return None


# for shot in range(dets_unpacked.shape[0]):
#   print(f'shot = {shot}')
shot = 0
errs_used = decode(dets_unpacked[shot], det_beam=20)
assert errs_used is not None
print(np.where(sampled_errs[shot])[0])
print(f'correct decoding: {(errs_used == sampled_errs[shot]).all()}')
delta = errs_used ^ sampled_errs[shot]
print(f'delta.sum() = {delta.sum()}')

reproduced_sampled_dets = np.zeros(dets_unpacked.shape[1], dtype=bool)
for ei in np.where(sampled_errs[shot])[0]:
  for d in edets[ei]:
    reproduced_sampled_dets[d] ^= True

reproduced_cost = np.dot(errs_used, ecosts)
reproduced_dets = np.zeros(dets_unpacked.shape[1], dtype=bool)
for ei in np.where(errs_used)[0]:
  for d in edets[ei]:
    reproduced_dets[d] ^= True

delta_dets = np.zeros(dets_unpacked.shape[1], dtype=bool)
for ei in np.where(delta)[0]:
  for d in edets[ei]:
    delta_dets[d] ^= True

assert (reproduced_sampled_dets == reproduced_dets).all()

assert (reproduced_dets == dets_unpacked[shot]).all()
print(reproduced_cost)
