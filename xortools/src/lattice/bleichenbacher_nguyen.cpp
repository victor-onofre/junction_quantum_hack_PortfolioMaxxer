#include "bleichenbacher_nguyen.hpp"

#include <cassert>
#include <unordered_set>

#include "NTL/LLL.h"
#include "NTL/ZZ.h"

void lattice_intersection_basis(const NTL::mat_ZZ& A, const NTL::mat_ZZ& B,
                                NTL::mat_ZZ& C) {
  // If the rows of A span an integer lattice LA and the rows of B span an
  // integer lattice LB, then the rows of the output matrix C will span an
  // integer lattice LA \cap LB
  // We compute this by preparing a matrix that looks like this:
  // [[A]
  //  [B]]
  // Let m1 = number of rows of A and m2 = number of rows of B.
  // We then use NTL::LLL to compute a basis for the left kernel of the above
  // matrix. The first m1 columns of this kernel basis, when multiplied by A,
  // give a basis for the intersection of A and B. Informally, this is because
  // these are the vectors in A that have some equivalent representation in B.
  NTL::mat_ZZ AoverB;
  // Ambient space dimension
  int n = A.NumCols();
  // size of basis A
  int m1 = A.NumRows();
  // size of basis B
  int m2 = B.NumRows();
  // assert(n >= m1);
  // assert(n >= m2);
  // Must have matching ambient dimensions
  assert(n == B.NumCols());
  AoverB.SetDims(m1 + m2, n);
  for (int i = 0; i < m1; ++i) {
    for (int j = 0; j < n; ++j) {
      AoverB[i][j] = A[i][j];
    }
  }
  for (int i = 0; i < m2; ++i) {
    for (int j = 0; j < n; ++j) {
      AoverB[m1 + i][j] = B[i][j];
    }
  }
  // Use LLL reduction to compute a multiple (the square) of the determinant
  // std::cout << "AoverB = " << AoverB << "\n";
  NTL::ZZ det2;
  NTL::mat_ZZ U;
  long rank = NTL::image(det2, AoverB, U, /*verbose=*/0);
  // std::cout << "got rank = " << rank << " det2 = " << det2 << "\n";
  long kernel_rank = m1 + m2 - rank;
  // std::cout << "U = " << U << "\n";
  // std::cout << "kernel_rank = " << kernel_rank << "\n";
  if (kernel_rank == 0) {
    C.SetDims(0, n);
    return;
  }
  assert(U.NumRows() == m1 + m2);
  assert(U.NumCols() == m1 + m2);
  NTL::mat_ZZ U_ker_trunc;
  U_ker_trunc.SetDims(kernel_rank, m1);
  for (int i = 0; i < kernel_rank; ++i) {
    for (int j = 0; j < m1; ++j) {
      U_ker_trunc[i][j] = U[i][j];
    }
  }
  NTL::mat_ZZ IntersectionGens;
  IntersectionGens.SetDims(kernel_rank, n);
  NTL::mul(IntersectionGens, U_ker_trunc, A);
  NTL::ZZ intersection_det2;
  long intersection_rank =
      NTL::image(intersection_det2, IntersectionGens, /*verbose=*/0);
  // std::cout <<"intersection_rank = "<<intersection_rank<<"\n";
  C.SetDims(intersection_rank, n);
  for (int i = 0; i < intersection_rank; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i][j] = IntersectionGens[kernel_rank - intersection_rank + i][j];
    }
  }
  // std::cout <<"C = "<<C<<"\n";
}

void get_same_sum_lattice_basis(const std::vector<std::vector<int>>& targets,
                                int total_targets_size, NTL::mat_ZZ& basis) {
  int m = targets.size();
  NTL::mat_ZZ same_sum_kernel_gens;
  assert(m > 0);
  same_sum_kernel_gens.SetDims(total_targets_size, m - 1);
  int offset = targets[0].size();
  for (int i = 1; i < m; ++i) {
    for (int j = 0; j < (int)targets[0].size(); ++j) {
      same_sum_kernel_gens[j][i - 1] = +1;
    }
    for (int j = 0; j < (int)targets[i].size(); ++j) {
      same_sum_kernel_gens[offset + j][i - 1] = -1;
    }
    offset += targets[i].size();
  }
  NTL::ZZ det2;
  NTL::mat_ZZ U;
  long rank = NTL::image(det2, same_sum_kernel_gens, U, /*verbose=*/0);
  long kernel_rank = total_targets_size - rank;
  assert(kernel_rank == total_targets_size + 1 - m);
  basis.SetDims(kernel_rank, total_targets_size);
  for (int i = 0; i < kernel_rank; ++i) {
    for (int j = 0; j < total_targets_size; ++j) {
      basis[i][j] = U[i][j];
    }
  }
}

LatticeAttackResult bleichenbacher_nguyen(
    const sparse::SparseMatrix& B, std::vector<std::vector<int>>& targets,
    bool verbose, bool use_lll, bool use_bkz, int max_block_size,
    bool apply_same_sum_constraint) {
  // Generate an NTL matrix for the generator matrix
  int m = B.ncols;
  NTL::ZZ p(B.modulus);
  NTL::ZZ_p::init(p);
  NTL::mat_ZZ_p G;

  // We take a transpose here
  G.SetDims(B.ncols, B.nrows);
  for (const sparse::SparseMatrix::Entry& entry : B.entries) {
    G[entry.col][entry.row] = entry.val;
  }
  if (verbose) {
    std::cout << "got generator matrix of dims (" << G.NumRows() << ", "
              << G.NumCols() << ")\n";
  }

  // Compute the parity check matrix
  NTL::mat_ZZ_p H;
  NTL::kernel(H, G);
  if (verbose) {
    std::cout << "got parity check matrix of dims (" << H.NumRows() << ", "
              << H.NumCols() << ")\n";
  }

  // Create the list matrix
  int total_targets_size = 0;
  for (int i = 0; i < m; ++i) {
    total_targets_size += targets[i].size();
  }
  NTL::mat_ZZ_p L;
  NTL::mat_ZZ Lz;
  L.SetDims(B.ncols, total_targets_size);
  Lz.SetDims(B.ncols, total_targets_size);
  int offset = 0;
  for (int i = 0; i < m; ++i) {
    for (int entry : targets[i]) {
      L[i][offset] = entry;
      Lz[i][offset] = entry;
      ++offset;
    }
  }
  if (verbose) {
    std::cout << "L has shape (" << L.NumRows() << ", " << L.NumCols() << ")\n";
  }

  // Multiply H * L to produce HL
  NTL::mat_ZZ_p HL;
  NTL::mul(HL, H, L);
  if (verbose) {
    std::cout << "HL has shape (" << HL.NumRows() << ", " << HL.NumCols()
              << ")\n";
  }

  // Get the right-kernel of HL, which includes all valid solutions (which are
  // low-weight binary vectors)
  NTL::mat_ZZ_p Delta;
  NTL::mat_ZZ_p HLT;
  NTL::transpose(HLT, HL);
  NTL::kernel(Delta, HLT);
  if (verbose) {
    std::cout << "Delta has shape (" << Delta.NumRows() << ", "
              << Delta.NumCols() << ")\n";
  }
  // Take Delta^T
  NTL::mat_ZZ_p DeltaT;
  NTL::transpose(DeltaT, Delta);

  // Append columns of the matrix Ip, and move to working over Z
  NTL::mat_ZZ DeltaTIp;
  DeltaTIp.SetDims(DeltaT.NumRows(), DeltaT.NumCols() + DeltaT.NumRows());
  for (int i = 0; i < DeltaTIp.NumRows(); ++i) {
    for (int j = 0; j < DeltaT.NumCols(); ++j) {
      DeltaTIp[i][j] = NTL::rep(DeltaT[i][j]);
    }
    DeltaTIp[i][i + DeltaT.NumCols()] = p;
  }

  NTL::mat_ZZ DeltaTIpT;
  NTL::transpose(DeltaTIpT, DeltaTIp);

  if (apply_same_sum_constraint) {
    NTL::mat_ZZ same_sum_basis;
    // Compute the intersection of the lattice spanned by the rows of DeltaTIpT
    // with the same-sum lattice.
    get_same_sum_lattice_basis(targets, total_targets_size, same_sum_basis);
    assert(DeltaTIpT.NumCols() == total_targets_size);
    assert(same_sum_basis.NumCols() == total_targets_size);
    // Compute the intersection of the row-span of DeltaTIpT with the row-span
    // of same_sum_basis
    NTL::mat_ZZ intersection_basis;
    lattice_intersection_basis(DeltaTIpT, same_sum_basis, intersection_basis);

    // Now, swap intersection_basis and DeltaTIpT
    std::swap(DeltaTIpT, intersection_basis);
    // And re-compute the un-transposed basis matrix
    NTL::transpose(DeltaTIp, DeltaTIpT);
  }

  if (verbose) {
    std::cout << "Original basis:" << "\n";
    std::cout << DeltaTIpT << "\n";
  }

  if (use_lll) {
    double delta = 0.99;
    NTL::LLL_FP(DeltaTIp, delta, /*deep=*/0, /*check=*/0,
                /*verbose=*/int(verbose));
  }
  if (use_bkz and DeltaTIpT.NumRows() > 1) {
    int block_size = DeltaTIpT.NumRows();
    if (max_block_size > 0) {
      block_size = std::min(max_block_size, block_size);
    }
    long prune = 0;
    double delta = 0.99;
    NTL::BKZ_FP(DeltaTIpT, delta, block_size, prune, /*check=*/0,
                /*verbose=*/int(verbose));
  }
  if (verbose) {
    std::cout << "Reduced basis: " << DeltaTIpT << "\n";
  }
  NTL::transpose(DeltaTIp, DeltaTIpT);

  // Check for valid solutions -- these are binary columns with exactly weight 1
  // in each of the blocks
  NTL::mat_ZZ LzDeltaTip;
  NTL::mul(LzDeltaTip, Lz, DeltaTIp);
  assert(LzDeltaTip.NumRows() == m);
  std::vector<std::unordered_set<int>> target_sets;
  for (int i = 0; i < m; ++i) {
    target_sets.push_back(
        std::unordered_set<int>(targets[i].begin(), targets[i].end()));
  }

  LatticeAttackResult result;
  // Each row gives two possible solutions depending on sign
  for (int j = 0; j < DeltaTIpT.NumRows(); ++j) {
    // std::cout << "DeltaTIpT[j] = " << DeltaTIpT[j] << "\n";
    // Compute the L1norm and L2normsquared of the solution vector
    size_t L1norm = 0;
    size_t L2norm2 = 0;
    for (int i = 0; i < DeltaTIpT.NumCols(); ++i) {
      assert(j < DeltaTIpT.NumRows());
      assert(i < DeltaTIpT.NumCols());
      int entry = NTL::conv<int>(DeltaTIpT[j][i]);
      L1norm += std::abs(entry);
      L2norm2 += entry * entry;
    }
    if (result.min_l2_squared == 0 or
        (L2norm2 > 0 and result.min_l2_squared > L2norm2)) {
      result.min_l2_squared = L2norm2;
    }
    if (result.min_l1 == 0 or (L1norm > 0 and result.min_l1 > L1norm)) {
      result.min_l1 = L1norm;
    }
    // if (L1norm == 1) {
    //   // std::cout << "solution with weight 1 found: ";
    //   for (int i = 0; i < DeltaTIpT.NumCols(); ++i) {
    //     int entry = NTL::conv<int>(DeltaTIpT[j][i]);
    //     // std::cout << entry << ", ";
    //   }
    //   // std::cout << "\n";
    // }

    // Try each sign
    for (int sign : {+1, -1}) {
      // Count up the number of satisfied constraints if we use this solution
      // vector multiplied by the list matrix
      int num_sat = 0;
      for (int i = 0; i < m; ++i) {
        int entry = NTL::conv<int>(LzDeltaTip[i][j]);
        int value = (sign * entry) % B.modulus;
        if (value < 0) {
          value += B.modulus;
          assert(value >= 0);
        }
        if (target_sets[i].count(value)) {
          ++num_sat;
        }
      }
      if (num_sat > result.best_num_sat) {
        result.best_num_sat = num_sat;
        // std::cout << "new best_num_sat = " << num_sat << "\n";
      }
    }
  }
  if (verbose) {
    std::cout << "got solution with num_sat = " << result.best_num_sat
              << " smallest L2 norm √" << result.min_l2_squared
              << " smallest L1 norm " << result.min_l1 << "\n";
  }
  return result;
}

unsigned power(unsigned base, unsigned exponent) {
  if (exponent == 0) return 1;

  unsigned half_pow = pow(base, exponent / 2);
  if (exponent % 2 == 0) {
    return half_pow * half_pow;
  } else {
    return base * half_pow * half_pow;
  }
}

void gen_planted_random_instance(sparse::SparseMatrix& B,
                                 std::vector<std::vector<int>>& targets, int n,
                                 int m, int num_distractors, int p,
                                 bool dont_add_zero, bool sort_targets,
                                 std::mt19937_64& rng, bool rs_code) {
  if (rs_code and p < m) {
    throw std::invalid_argument("for rs_code=true mode, must have p >= m");
  }
  if (n > m) {
    throw std::invalid_argument("must have n <= m");
  }
  // Define a Reed-Solomon code generator matrix.
  B.resize(n, m);
  B.modulus = p;
  std::vector<std::vector<int>> B_vec(n);
  std::uniform_int_distribution<int> dis;
  for (int i = 0; i < n; ++i) {
    B_vec[i].resize(m);
    for (int j = 0; j < m; ++j) {
      sparse::SparseMatrix::Entry entry;
      entry.row = i;
      entry.col = j;
      if (rs_code) {
        // Use a Reed-Solomon Code B matrix
        entry.val = power(j, i) % p;
      } else {
        // Use a random B matrix
        entry.val = dis(rng) % p;
        if (entry.val < 0) {
          entry.val += p;
          assert(entry.val >= 0);
        }
      }
      if (entry.val != 0) {
        B.add_entry(entry);
      }
      B_vec[i][j] = entry.val;
    }
  }
  targets.clear();
  targets.resize(m);
  std::vector<int> planted_solution(n);
  for (int i = 0; i < n; ++i) {
    planted_solution[i] = dis(rng) % p;
  }
  std::vector<int> planted_string(m, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      planted_string[j] += planted_solution[i] * B_vec[i][j];
      planted_string[j] %= p;
      if (planted_string[j] < 0) {
        planted_string[j] += p;
        assert(planted_string[j] >= 0);
      }
    }
  }
  // Add the planted string and the distractor points to the targets
  if (num_distractors + 1 > p) {
    throw std::invalid_argument(
        "Must have num_distractors < p so that num_distractors + 1 <= p");
  }
  if (dont_add_zero and num_distractors + 2 > p) {
    throw std::invalid_argument(
        "Since dont_add_zero was passed, we must have num_distractors < p - "
        "1.");
  }
  for (int j = 0; j < m; ++j) {
    targets[j].push_back(planted_string[j]);
    std::unordered_set<int> forbidden;

    // Add forbidden elements to the set
    if (dont_add_zero) {
      forbidden.insert(0);
    }
    forbidden.insert(planted_string[j]);

    // Generate distractors
    while (targets[j].size() < size_t(num_distractors + 1)) {
      std::uniform_int_distribution<int> distribution(0, p - 1);
      int candidate = distribution(rng);

      if (forbidden.find(candidate) != forbidden.end()) {
        continue;
      }
      targets[j].push_back(candidate);
      forbidden.insert(candidate);
    }
    if (sort_targets) {
      std::sort(targets[j].begin(), targets[j].end());
    }
  }
}
