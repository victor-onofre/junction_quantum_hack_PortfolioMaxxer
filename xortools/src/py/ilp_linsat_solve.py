import argparse
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, OptimizeResult



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



def main(csv_file):
    print(f"Attacking ISIS_∞ LINSAT instance from file: {csv_file}")
    instance = load_instance_from_csv(csv_file)

    p = instance['p']
    m = instance['m']
    constraints = instance['constraints']
    n = max(max(c['variables'], default=-1) for c in constraints) + 1  # number of variables
    y = np.random.randint(p, size=(m,))
    beta = int(p * 0.3)
    # beta = 1

    num_vars = n + m  # x_0, ..., x_{n-1}, s_0, ..., s_{m-1}

    # Objective: We don't care about optimizing any function, so we set it to zero.
    c = np.zeros(num_vars)

    # Build constraint matrix A and bounds
    A = []
    lb = []
    ub = []

    for i, constraint in enumerate(constraints):
        row = np.zeros(num_vars)
        for var_idx, coeff in zip(constraint['variables'], constraint['coefficients']):
            row[var_idx] = coeff
        row[n + i] = p  # Slack variable coefficient
        A.append(row)
        lb.append(y[i] - beta)
        ub.append(y[i] + beta)

    A = np.array(A)
    constraint = LinearConstraint(A, lb, ub)

    # Variable bounds: x_i in [0, p-1], s_i unrestricted
    x_bounds = [(0, p - 1)] * n
    s_bounds = [(-np.inf, np.inf)] * m
    bounds = Bounds([b[0] for b in x_bounds + s_bounds],
                    [b[1] for b in x_bounds + s_bounds])

    # Integer constraints: All variables are integers
    integrality = np.ones(num_vars, dtype=bool)

    # Solve the MILP
    result: OptimizeResult = milp(c=c, constraints=[constraint],
                                  integrality=integrality, bounds=bounds)

    if result.success:
        x = result.x[:n].round().astype(int)
        print("Found a valid solution x:")
        print(x)
    else:
        print("MILP failed to find a solution.")
        print(result)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="Attack an instance of exact ISIS_∞ LINSAT using integer linear programming"
  )
  parser.add_argument(
      "csv_file",
      type=str,
      help="Path to the CSV file describing the ISIS_∞ LINSAT instance"
  )

  args = parser.parse_args()
  main(args.csv_file)