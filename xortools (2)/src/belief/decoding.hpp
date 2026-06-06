#ifndef BELIEF_DECODING_HPP
#define BELIEF_DECODING_HPP

#include "bitmatrix.hpp"
#include "sparse/sparse_matrix.hpp"

bitmatrix parity_check_bitmatrix(const sparse::SparseMatrix& H);

double binary_entropy(double p);

double inverse_binary_entropy(double rate, double tolerance = 1e-6);

size_t maximum_possible_ell(double num_rows, double num_cols);

double get_max_tolerable_error_rate(const sparse::SparseMatrix& B,
                                    double max_decoding_error_rate,
                                    size_t shots_per_point, int maxiter,
                                    size_t num_threads, std::mt19937_64& eng,
                                    bool verbose);

#endif  // BELIEF_DECODING_HPP
