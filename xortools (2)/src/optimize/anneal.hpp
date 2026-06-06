#ifndef __ANNEAL_HPP__
#define __ANNEAL_HPP__

#include <vector>

#include "sparse/sparse_matrix.hpp"

int evaluate(const sparse::SparseMatrix& H, const std::vector<int>& target,
             const std::vector<bool>& x);

std::vector<bool> anneal(const sparse::SparseMatrix& H,
                         const std::vector<int>& target, size_t iterations,
                         double min_beta, double max_beta,
                         double min_acceptance_rate,
                         std::function<uint64_t(void)> gen,
                         bool verbose = false);

// x is the initial solution to start from for annealing
std::vector<bool> anneal(const sparse::SparseMatrix& H,
                         const std::vector<int>& target, std::vector<bool>& x,
                         size_t iterations, double min_beta, double max_beta,
                         double min_acceptance_rate,
                         std::function<uint64_t(void)> gen,
                         bool verbose = false);

std::vector<bool> irreg_anneal(const sparse::SparseMatrix& H,
                               const std::vector<int>& target,
                               size_t iterations,
                               std::function<uint64_t(void)> gen, bool verbose);

#endif  // __ANNEAL_HPP__
