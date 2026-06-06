import numpy as np
from bp import load_instance_from_csv
import itertools


def objective(instance, sol, subset_cons=None):
  if subset_cons is None:
    subset_cons = set(range(instance['m']))

  num_sat = 0
  for i in subset_cons:
    cons = instance['constraints'][i]
    total = sum(
      c * sol[v]
      for c, v in zip(cons['coefficients'], cons['variables'])
    ) % instance['p']
    num_sat += (total in cons['sum_values'])
  return num_sat

def all_connected_subgraphs(g, m):
  def _recurse(t, possible, excluded):
    if len(t) == m:
      yield t
    else:
      excluded = set(excluded)
      for i in possible:
        if i not in excluded:
          new_t = (*t, i)
          new_possible = possible | g[i]
          excluded.add(i)
          yield from _recurse(new_t, new_possible, excluded)
  excluded = set()
  for (i, possible) in enumerate(g):
    excluded.add(i)
    yield from _recurse((i,), possible, excluded)

def find_improvement(instance, sol, t=1, return_early=False):
  # Looks at all strings in a Hamming ball of distance t
  # for an improvement
  current = objective(instance, sol)
  n = instance['n']
  p = instance['p']
  best_change = np.zeros(n, dtype=int)
  best_obj = current
  for r in range(1, t+1):
    # for coords in itertools.combinations(range(n), r=r):
    for coords in all_connected_subgraphs(instance['variable_graph'], r):
      coords = np.array(coords, dtype=int)
      objs = set()
      for v in coords:
        objs = objs.union(instance['variable_to_constraints'][v])
      current_value_on_objs = objective(instance, sol, objs)
      if current_value_on_objs == len(objs):
        # If all of these objectives are already satisfied, then
        # skip
        continue
      for vals in itertools.product(range(1, p), repeat=r):
        change = np.zeros(n, dtype=int)
        change[coords] = vals
        new_val_on_objs = objective(instance, sol + change, objs)
        new_obj = current - current_value_on_objs + new_val_on_objs
        if new_obj > best_obj:
          best_change = change
          best_obj = new_obj
          if return_early:
            return ((sol + best_change) % instance['p'], best_obj)
        # If it is not possible to find a better improvement
        # with these coordinates, then quit looking
        if best_obj - current >= len(objs) - current_value_on_objs:
          break
  return ((sol + best_change) % instance['p'], best_obj)

def greedy(instance, t=1, use_return_early=False):
  p = instance['p']
  n = instance['n']
  m = instance['m']
  sol = np.random.randint(0, p, size=(n,))
  obj = objective(instance, sol)
  print(f'initial objective: {obj} clauses satisfied out of {m}')
  while True:
    better_sol, better_obj = find_improvement(
      instance, sol, t=t, return_early=use_return_early)
    if better_obj == obj:
      assert objective(instance, better_sol) == obj
      return sol, obj
    sol = better_sol
    obj = better_obj
    print(f'improved to {obj}')

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser('do a greedy search')
  parser.add_argument('--csv', required=True, type=str)
  parser.add_argument('--t', required=False, default=1, type=int,
                      help='max size of a subset of nodes to consider changing')
  parser.add_argument(
    '--use-return-early',
    help='whether to use early return on improvement search',
    action=argparse.BooleanOptionalAction,
    default=True)
  args = parser.parse_args()

  instance = load_instance_from_csv(args.csv)

  # Create a lookup table of which constraints each variable is in
  instance['variable_to_constraints'] = [[] for v in range(instance['n'])]
  for i, cons in enumerate(instance['constraints']):
    for v in cons['variables']:
      instance['variable_to_constraints'][v].append(i)

  # Create a graph on the variables where edges indicate a common clause
  instance['variable_graph'] = [set() for v in range(instance['n'])]
  for v1 in range(instance['n']):
    for i in instance['variable_to_constraints'][v1]:
      for v2 in instance['constraints'][i]['variables']:
        if v1 != v2:
          instance['variable_graph'][v2].add(v1)
          instance['variable_graph'][v1].add(v2)

  sol, obj = greedy(instance, t=args.t, use_return_early=args.use_return_early)
  print(f'solution: {",".join(map(str, sol))}')
  print(f'{obj} clauses satisfied out of {instance["m"]}')
