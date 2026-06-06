from ortools.sat.python import cp_model
from pysat.formula import WCNF

def solve_maxsat_from_wcnf(filename):
    wcnf = WCNF(from_file=filename)

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # Variables
    variables = {}
    for var in range(1, wcnf.nv + 1):
        variables[var] = model.NewBoolVar(f"x_{var}")
        variables[-var] = variables[var].Not()  # Negation

    # Hard clauses
    for clause in wcnf.hard:
        model.AddBoolOr([variables[lit] for lit in clause])

    # Soft clauses and objective
    total_weight_satisfied = model.NewIntVar(0, len(wcnf.soft), "total_weight_satisfied")
    weighted_satisfied_clauses = []
    for clause in wcnf.soft:
        soft_clause_sat = model.NewBoolVar(f"soft_sat_{clause}")
        model.AddBoolOr([variables[lit] for lit in clause] + [soft_clause_sat.Not()])
        weighted_satisfied_clauses.append(soft_clause_sat)

    model.Add(total_weight_satisfied == sum(weighted_satisfied_clauses))

    # Maximize
    model.Maximize(total_weight_satisfied)
    status = solver.Solve(model)
    print(status)
    
    if status == cp_model.OPTIMAL:
        print("Optimal solution found:")
        for var in range(1, wcnf.nv + 1):
            print(f"x_{var} = {solver.Value(variables[var])}")
    else:
        print("No optimal solution found.")

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    solve_maxsat_from_wcnf(filename)
