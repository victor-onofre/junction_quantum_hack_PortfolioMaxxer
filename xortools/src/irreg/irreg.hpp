#ifndef IRREG_IRREG_HPP
#define IRREG_IRREG_HPP

#include <functional>
#include <vector>

#include "irreg/distribution.hpp"
#include "sparse/sparse_matrix.hpp"

// Samples an LDPC code parity check matrix (Tanner graph) represented as a
// sparse matrix from the given edge-perspective degree distribution pair. No
// effort is made to maximize the girth.
sparse::SparseMatrix sample_irreg(int m, int n,
                                  const DegreeDistributionPair& dist,
                                  std::function<unsigned(void)> gen,
                                  bool verbose);

#endif  // IRREG_IRREG_HPP
