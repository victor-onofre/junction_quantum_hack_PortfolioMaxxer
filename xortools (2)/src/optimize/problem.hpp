#ifndef OPTIMIZE_PROBLEM_HPP
#define OPTIMIZE_PROBLEM_HPP

#include "sparse/sparse_matrix.hpp"

struct MaxXORSATProblem {
  sparse::SparseMatrix H;
  std::vector<int> targets;
};

void strip_high_degree_clauses(const MaxXORSATProblem& problem,
                               std::map<int, MaxXORSATProblem>& subproblems,
                               bool verbose);

#endif  // OPTIMIZE_PROBLEM_HPP
