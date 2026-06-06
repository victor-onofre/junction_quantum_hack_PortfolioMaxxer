#ifndef DESIGN_DESIGN_HPP
#define DESIGN_DESIGN_HPP

#include <cstdint>

#include "irreg/distribution.hpp"

double est_dqival(double delta);

double est_thval(const DegreeDistributionPair& dist);

double est_saval(double D);

double est_adaptive_saval_model(DegreeDistributionPair dist);

double est_saval_empirical(DegreeDistributionPair dist, size_t num_clauses,
                           size_t num_samples, size_t num_iterations,
                           uint64_t seed, std::vector<double>& values);

double est_stripped_saval_empirical(DegreeDistributionPair dist,
                                    size_t num_clauses, size_t num_samples,
                                    size_t num_iterations, uint64_t seed);

double est_quantum_advantage_over_sa_empirical(
    DegreeDistributionPair dist, size_t num_clauses, size_t num_samples,
    size_t num_iterations, double max_error_rate, int max_iter,
    size_t shots_per_point, uint64_t seed);

double est_adaptive_saval_empirical(DegreeDistributionPair dist, uint64_t seed);

double est_decoding_threshold_empirical(DegreeDistributionPair dist,
                                        size_t num_nodes, size_t num_samples,
                                        double max_error_rate, int max_iter,
                                        size_t shots_per_point, uint64_t seed);

#endif  // DESIGN_DESIGN_HPP
