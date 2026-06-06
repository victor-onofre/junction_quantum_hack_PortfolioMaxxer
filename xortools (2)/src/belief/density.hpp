#ifndef BELIEF_DENSITY_HPP
#define BELIEF_DENSITY_HPP
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "irreg/distribution.hpp"

inline void zfill(std::vector<double>& vec) {
  std::fill(vec.begin(), vec.end(), 0);
};

inline bool is_close(const std::vector<double>& v1,
                     const std::vector<double>& v2, double epsilon = 1e-4) {
  assert(v1.size() == v2.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    if (std::abs(v1[i] - v2[i]) > epsilon) {
      std::cerr << "v1[" << i << "] = " << v1[i] << " v2[" << i
                << "] = " << v2[i] << "\n";
      return false;
    }
  }
  return true;
}

inline void renormalize(std::vector<double>& vec) {
  double sum = 0;
  for (const double& elem : vec) sum += elem;
  assert(sum > 0);
  for (double& elem : vec) elem /= sum;
}

// Allow for up to 2 different bins to be populated as a result of this op.
struct QuantizeResult {
  uint16_t i1;
  double p1;
  uint16_t i2;
  double p2;
};

// Leverages the caches for fast xor and product computation
inline void fast_op(const std::vector<double>& d1,
                    const std::vector<double>& d2, std::vector<double>& out,
                    const std::vector<QuantizeResult>& flat_op_cache,
                    size_t num_llr_bins) noexcept {
  for (size_t i = 0; i < num_llr_bins; ++i) {
    if (d1[i] == 0) continue;
    for (size_t j = 0; j < num_llr_bins; ++j) {
      if (d2[j] == 0) continue;
      const QuantizeResult& qr = flat_op_cache[num_llr_bins * i + j];
      out[qr.i1] += d1[i] * d2[j] * qr.p1;
      out[qr.i2] += d1[i] * d2[j] * qr.p2;
    }
  }
}

std::pair<std::vector<QuantizeResult>, std::vector<QuantizeResult>>
get_quantized_tables(const std::vector<double>& quantized_llrs, double min_llr,
                     double max_llr, size_t num_llr_bins,
                     bool fractional_quantization);

void linspace(std::vector<double>& out, double min, double max, size_t bins);

QuantizeResult quantize(const double llr, double min_llr, double max_llr,
                        size_t num_llr_bins,
                        const std::vector<double>& quantized_llrs,
                        bool fractional);

void bintree_update(const DegreeDistribution& dist,
                    std::vector<double>& in_density,
                    std::vector<double>& out_density,
                    const std::vector<QuantizeResult>& op_cache,
                    size_t num_llr_bins, bool is_last_iteration_node_update);

// Returns true if it looks to be below threshold after the number of iterations
bool density_evolution(
    const DegreeDistributionPair& dist, double delta, size_t num_llr_bins,
    size_t num_iterations, bool fractional_quantization, bool early_stop,
    bool quantization_fiddling, double shrink_factor, bool verbose,
    std::function<void(const std::vector<double>&)> density_callback =
        [](const std::vector<double>& density) {});

// Binary search to find the threshold down to a given precision.
// If halt_if_leq > 0, then as soon as the threshold is determined to be <=
// halt_if_leq, the search concludes.
std::pair<double, double> binary_search_threshold(
    const DegreeDistributionPair& dist, double delta_min, double delta_max,
    double precision, size_t num_llr_bins, size_t num_iterations,
    bool fractional_quantization, bool early_stop, bool quantization_fiddling,
    double shrink_factor, bool verbose, double halt_if_leq);

#endif  // BELIEF_DENSITY_HPP
