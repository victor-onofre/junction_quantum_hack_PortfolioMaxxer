#ifndef BLEICHENBACHER_NGUYEN_HPP
#define BLEICHENBACHER_NGUYEN_HPP

#include <vector>

#include "NTL/ZZ.h"
#include "NTL/mat_ZZ.h"
#include "sparse/sparse_matrix.hpp"

void lattice_intersection_basis(const NTL::mat_ZZ& A, const NTL::mat_ZZ& B,
                                NTL::mat_ZZ& C);

struct LatticeAttackResult {
  int min_l2_squared = 0;
  int min_l1 = 0;
  int best_num_sat = 0;
};

LatticeAttackResult bleichenbacher_nguyen(
    const sparse::SparseMatrix& B, std::vector<std::vector<int>>& targets,
    bool verbose, bool use_lll, bool use_bkz, int max_block_size,
    bool apply_same_sum_constraint);

void gen_planted_random_instance(sparse::SparseMatrix& B,
                                 std::vector<std::vector<int>>& targets, int n,
                                 int m, int num_distractors, int p,
                                 bool dont_add_zero, bool sort_targets,
                                 std::mt19937_64& rng, bool rs_code);
#endif  // BLEICHENBACHER_NGUYEN_HPP
