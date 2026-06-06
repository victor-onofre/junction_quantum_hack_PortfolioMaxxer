#include "problem.hpp"

#include <set>

void strip_high_degree_clauses(const MaxXORSATProblem& problem,
                               std::map<int, MaxXORSATProblem>& subproblems,
                               bool verbose) {
  std::map<size_t, std::vector<size_t>> clauses_by_degree;
  for (size_t i = 0; i < problem.H.ncols; ++i) {
    // clause i has degree H.col_entries[i].size()
    clauses_by_degree[problem.H.col_entries[i].size()].push_back(i);
  }
  std::set<size_t> allowed_clauses;
  for (const auto& [degree, clauses] : clauses_by_degree) {
    for (size_t i : clauses) {
      allowed_clauses.insert(i);
    }
    if (verbose) {
      std::cout << "if we strip clauses of degree > " << degree
                << ", there are " << allowed_clauses.size()
                << " out of the original " << problem.H.ncols << " clauses.\n";
    }
    sparse::SparseMatrix H_stripped(problem.H.nrows, allowed_clauses.size());
    std::vector<int> target_stripped;
    size_t i_stripped = 0;
    for (size_t i : allowed_clauses) {
      for (const auto& it : problem.H.col_entries[i]) {
        const size_t& j = it.first;
        sparse::SparseMatrix::Entry entry;
        entry.col = i_stripped;
        entry.row = j;
        entry.val = 1;
        H_stripped.add_entry(entry);
      }
      target_stripped.push_back(problem.targets[i]);
      ++i_stripped;
    }
    subproblems[degree] = {H_stripped, target_stripped};
  }
}
