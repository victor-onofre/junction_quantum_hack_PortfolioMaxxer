#include "belief/density.hpp"

#include <cassert>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "density.hpp"
#include "utils.hpp"

std::pair<std::vector<QuantizeResult>, std::vector<QuantizeResult>>
get_quantized_tables(const std::vector<double>& quantized_llrs, double min_llr,
                     double max_llr, size_t num_llr_bins,
                     bool fractional_quantization) {
  // Populate quantized XOR cache and Product cache. Note that both of these
  // operations are associative.
  std::vector<QuantizeResult> xor_cache(num_llr_bins * num_llr_bins);
  std::vector<QuantizeResult> prod_cache((num_llr_bins * num_llr_bins));
  for (size_t i = 0; i < num_llr_bins; ++i) {
    double llr_p = quantized_llrs[i];
    double p_over_one_minus_p = std::exp(llr_p);
    double p = p_over_one_minus_p / (1 + p_over_one_minus_p);
    for (size_t j = 0; j < num_llr_bins; ++j) {
      double llr_q = quantized_llrs[j];
      double q_over_one_minus_q = std::exp(llr_q);
      double q = q_over_one_minus_q / (1 + q_over_one_minus_q);
      // "Product" check update rule
      {
        double w = p * (1 - q) + (1 - p) * q;
        double llr = std::log(w / (1 - w));
        xor_cache[num_llr_bins * i + j] =
            quantize(llr, min_llr, max_llr, num_llr_bins, quantized_llrs,
                     fractional_quantization);
      }
      // "Sum" node update rule
      {
        double llr = llr_p + llr_q;
        prod_cache[num_llr_bins * i + j] =
            quantize(llr, min_llr, max_llr, num_llr_bins, quantized_llrs,
                     fractional_quantization);
      }
    }
  }
  return {xor_cache, prod_cache};
}

void linspace(std::vector<double>& out, double min, double max, size_t bins) {
  assert(bins > 0);
  for (size_t i = 0; i < bins; ++i) {
    out.push_back(double(i) / double(bins - 1) * (max - min) + min);
  }
}

QuantizeResult quantize(const double llr, double min_llr, double max_llr,
                        size_t num_llr_bins,
                        const std::vector<double>& quantized_llrs,
                        bool fractional) {
  QuantizeResult result;
  // Handle outside range as a special case.
  if (llr <= min_llr) {
    result.i1 = result.i2 = 0;
    result.p1 = 1.0;
    result.p2 = 0.0;
    return result;
  } else if (llr >= max_llr) {
    result.i1 = result.i2 = num_llr_bins - 1;
    result.p1 = 1.0;
    result.p2 = 0.0;
    return result;
  }

  // Calculate the proportional position of llr
  double proportion = (llr - min_llr) / (max_llr - min_llr);
  assert(proportion >= 0 && proportion <= 1);

  // Calculate the tentative index
  double tentative_index = proportion * (num_llr_bins - 1);
  assert(tentative_index >= 0);
  size_t lower_index = std::floor(tentative_index);

  // Ensure it's within bounds
  assert(num_llr_bins > 2);
  lower_index = std::max<size_t>(0, std::min(lower_index, num_llr_bins - 2));
  size_t upper_index = lower_index + 1;

  if (fractional) {
    double lower_value = quantized_llrs[lower_index];
    double upper_value = quantized_llrs[upper_index];

    // Compute weights
    result.p1 = (upper_value - llr) / (upper_value - lower_value);
    result.p2 = 1.0 - result.p1;

    result.i1 = lower_index;
    result.i2 = upper_index;
  } else {
    // Non-fractional behavior: find the closest index
    size_t closest_index = static_cast<size_t>(std::round(tentative_index));
    closest_index =
        std::max<size_t>(0, std::min(closest_index, num_llr_bins - 1));
    result.i1 = result.i2 = closest_index;
    result.p1 = 1.0;
    result.p2 = 0.0;
  }

  return result;
}

struct ReverseCache {
  std::vector<size_t> offsets;
  std::vector<std::pair<uint16_t, uint16_t>> ijs;
  std::vector<std::tuple<uint16_t, uint16_t, uint16_t>> ijks;
  uint16_t num_llr_bins;
  ReverseCache(const std::vector<uint16_t>& flat_op_cache,
               size_t num_llr_bins) noexcept
      : num_llr_bins(num_llr_bins) {
    std::vector<std::vector<std::pair<uint16_t, uint16_t>>> out_to_ij(
        num_llr_bins);
    for (size_t i = 0; i < num_llr_bins; ++i) {
      for (size_t j = 0; j < num_llr_bins; ++j) {
        out_to_ij[flat_op_cache[num_llr_bins * i + j]].push_back({i, j});
      }
    }
    size_t offset = 0;
    for (uint16_t out = 0; out < num_llr_bins; ++out) {
      offsets.push_back(offset);
      for (const auto& ij : out_to_ij[out]) {
        ijs.push_back(ij);
        ijks.push_back({ij.first, ij.second, out});
        ++offset;
      }
    }
  }
  inline void fast_op(std::vector<double>& d1, std::vector<double>& d2,
                      std::vector<double>& result) noexcept {
    for (size_t z = 0; z < ijks.size(); ++z) {
      const auto& [i, j, k] = ijks[z];
      result[k] += d1[i] * d2[j];
    }
  }
  // inline void fast_op(std::vector<double>& d1, std::vector<double>& d2,
  //                   std::vector<double>& result) noexcept {

  //   for (size_t out=0; out<num_llr_bins; ++out) {
  //     size_t start = offsets[out];
  //     size_t end = ijs.size();
  //     if (out + 1 < num_llr_bins) end = offsets[out+1];
  //     double sum = 0;
  //     for (size_t offset = start; offset<end; ++offset) {
  //       const auto& [i,j] = ijs[offset];
  //       sum += d1[i]*d2[j];
  //     }
  //     result[out] = sum;
  //   }
  // }
};

// inline void fast_xor(std::vector<double>& d1, std::vector<double>& d2,
//                     std::vector<double>& out, size_t num_llr_bins) {

//   for (size_t i=0; i<num_llr_bins; ++i) {
//     for (size_t j=0; j < num_llr_bins; ++j) {

//     }
//   }
// }

void greedy_update(const DegreeDistribution& dist,
                   std::vector<double>& in_density,
                   std::vector<double>& out_density,
                   std::vector<QuantizeResult>& op_cache, size_t num_llr_bins) {
  std::vector<double> tmp(num_llr_bins, 0);
  // There are D-1 incoming messages to a check / node of degree D that get
  // passed out on any given wire. We initialze with 1 factor of the input
  // density, so d = 1;
  std::vector<double> d_fold_op_density = in_density;
  std::map<size_t, std::vector<double>> d_fold_densities;
  size_t d = 1;
  size_t index = 0;
  while (dist.degrees[index] < 2) {
    // Skip nodes of degree 0 and 1. Check nodes of degree 1 correspond to a
    // modified initial density.
    ++index;
  }
  while (true) {
    d_fold_densities[d] = d_fold_op_density;
    if (d == dist.degrees[index] - 1) {
      for (size_t i = 0; i < num_llr_bins; ++i) {
        out_density[i] += d_fold_op_density[i] * dist.probabilities[index];
      }
      ++index;
    }
    if (index == dist.degrees.size()) break;
    size_t d_lookup = dist.degrees[index] - 1 - d;
    assert(d_lookup > 0);
    // Greedy approach
    for (; d_lookup > 0; --d_lookup) {
      if (d_fold_densities.count(d_lookup)) break;
    }
    assert(d_fold_densities.count(d_lookup));
    // Add another copy of the input density to the xor-sum and increment d
    fast_op(d_fold_op_density, d_fold_densities.at(d_lookup), tmp, op_cache,
            num_llr_bins);
    std::swap(d_fold_op_density, tmp);
    zfill(tmp);
    d += d_lookup;
  }
  d_fold_densities.clear();
  // Renormalize out_density to help deal with numerical precision
  // issues
  renormalize(out_density);
}

inline size_t get_msb(size_t val) {
  size_t msb = 0;
  while (val >>= 1) ++msb;
  return msb;
}

inline size_t get_lsb(size_t val) {
  size_t lsb = 0;
  while (!(val & (1 << lsb))) ++lsb;
  return lsb;
}

// This cached binary tree update method is optimized for a mix of precision and
// speed. This should obey context-invariance of degree distributions, i.e. that
// the threshold is unaffected by the zero-probability entries of the
// distribution. Note that this is a suboptimal method. The optimal method would
// make use of minimum-size addition chains. However it is somehwat involved
// (though not known to be NP-hard) to compute the smallest addition chain and
// we also want the context-invariance property to hold.
void bintree_update(const DegreeDistribution& dist,
                    std::vector<double>& in_density,
                    std::vector<double>& out_density,
                    const std::vector<QuantizeResult>& op_cache,
                    size_t num_llr_bins, bool is_last_iteration_node_update) {
  // First step: compute a bunch of powers of 2
  size_t max_degree = dist.degrees.back();
  if (is_last_iteration_node_update) max_degree += 1;
  size_t msb = get_msb(max_degree - 1);
  std::vector<std::vector<double>> powers(msb + 1);
  powers[0] = in_density;
  for (size_t i = 1; i < msb + 1; ++i) {
    powers[i].resize(num_llr_bins);
    fast_op(powers[i - 1], powers[i - 1], powers[i], op_cache, num_llr_bins);
  }

  std::vector<double> tmp1, tmp2;
  tmp1.resize(num_llr_bins);
  tmp2.resize(num_llr_bins);

  std::map<size_t, std::vector<double>> prefixes;
  // Second step: compute all degrees using their binary representation.
  for (size_t i = 0; i < dist.degrees.size(); ++i) {
    if (dist.degrees[i] < 2) {
      // Skip nodes of degree 0 and 1. Check nodes of degree 1 are wasted; check
      // nodes of degree 1 correspond to a modified initial density.
      continue;
    }

    size_t d = dist.degrees[i] - 1;
    if (is_last_iteration_node_update) d += 1;
    // Find the most significant bit of d to which we have cached the
    // intermediate result
    size_t prefix_match = msb;
    size_t d_mask = d;
    while (prefix_match > 0 and !prefixes.count(d_mask)) {
      // Zero out this bit of the mask
      d_mask &= ~(1 << prefix_match--);
    }
    size_t b = prefix_match;
    if (prefix_match > 0) {
      tmp1 = prefixes.at(d_mask);
    } else {
      b = get_lsb(d);
      tmp1 = powers[b];
    }
    ++b;
    for (; b <= msb; ++b) {
      if (!(d & (1 << b))) continue;
      fast_op(tmp1, powers[b], tmp2, op_cache, num_llr_bins);
      std::swap(tmp1, tmp2);
      zfill(tmp2);
      prefixes[d_mask |= (1 << b)] = tmp1;
    }
    // Now we have prepared into tmp1 the degree-1 power of the in_density
    // We just have to add the proportionate amount to the out_density
    for (size_t j = 0; j < num_llr_bins; ++j) {
      out_density[j] += tmp1[j] * dist.probabilities[i];
    }
    zfill(tmp1);
  }
  // Renormalize out_density to help deal with numerical precision issues
  renormalize(out_density);
}

void check_trivial_nothreshold(const DegreeDistributionPair& dist) {
  for (size_t i = 0; i < dist.node.degrees.size(); ++i) {
    if (dist.node.degrees[i] == 0 and dist.node.probabilities[i] > 0) {
      throw std::invalid_argument(
          "There is no threshold under uniform bit-flip noise for this code "
          "because some nodes have degree 0 and are therefore incorrectible.");
    }
  }
}

// Returns true if it looks to be below threshold (i.e. the bit error rate is
// less than shrink_factor * initial error rate) after the number of iterations
bool density_evolution(
    const DegreeDistributionPair& dist, double delta, size_t num_llr_bins,
    size_t num_iterations, bool fractional_quantization, bool early_stop,
    bool quantization_fiddling, double shrink_factor, bool verbose,
    std::function<void(const std::vector<double>&)> density_callback) {
  if (shrink_factor >= 1 or shrink_factor < 0) {
    throw std::invalid_argument("shrink_factor must be < 1 and >= 0. Got " +
                                std::to_string(shrink_factor));
  }
  dist.assert_valid();
  check_trivial_nothreshold(dist);
  if (num_llr_bins % 2 != 1) {
    num_llr_bins += 1;
  }
  assert((delta > 0.0) && (delta < 0.5));
  double base_llr = std::log(delta / (1 - delta));
  assert(base_llr < 0);
  std::vector<double> quantized_llrs;
  double min_llr = -25;
  if (quantization_fiddling) {
    // Modify the min_llr and num_llr_bins slightly in order to guarantee
    // that we can exactly represent the quantized delta and 1-delta llr values.
    int scale_1 = std::ceil(25.0 / -base_llr);
    double new_min_llr = scale_1 * base_llr;
    int scale_2 = double(num_llr_bins) / 2.0 / scale_1;
    size_t new_num_llr_bins = 2 * scale_1 * scale_2 + 1;
    num_llr_bins = new_num_llr_bins;
    min_llr = new_min_llr;
  }
  double max_llr = -min_llr;
  assert(num_llr_bins >= 2);
  linspace(quantized_llrs, min_llr, max_llr, num_llr_bins);

  density_callback(quantized_llrs);

  double eps = 0;
  auto [xor_cache, prod_cache] = get_quantized_tables(
      quantized_llrs, min_llr, max_llr, num_llr_bins, fractional_quantization);
  // Prepare the initial density i.e. the distribution over the first messages
  // in BP
  std::vector<double> initial_density(num_llr_bins, 0);
  // TODO: accomodate degree 1 checks
  if (dist.check.degrees[0]) {
    QuantizeResult bq = quantize(base_llr, min_llr, max_llr, num_llr_bins,
                                 quantized_llrs, fractional_quantization);
    initial_density[bq.i1] += (1 - delta) * (1 - eps) * bq.p1;
    initial_density[bq.i2] += (1 - delta) * (1 - eps) * bq.p2;
    bq = quantize(-base_llr, min_llr, max_llr, num_llr_bins, quantized_llrs,
                  fractional_quantization);
    initial_density[bq.i1] += delta * (1 - eps) * bq.p1;
    initial_density[bq.i2] += delta * (1 - eps) * bq.p2;

    // Mix in some erasure errors
    bq = quantize(0, min_llr, max_llr, num_llr_bins, quantized_llrs,
                  fractional_quantization);
    initial_density[bq.i1] += eps * bq.p1;
    initial_density[bq.i2] += eps * bq.p2;
  }

  std::vector<double> density = initial_density;
  std::vector<double> check_out_density(num_llr_bins, 0);
  std::vector<double> node_out_density(num_llr_bins, 0);
  std::vector<double> d_fold_op_density(num_llr_bins, 0);
  std::vector<double> tmp(num_llr_bins, 0);
  density_callback(density);
  for (size_t t = 0; t < num_iterations; ++t) {
    // Compute distribution over check output messages
    bintree_update(dist.check, density, check_out_density, xor_cache,
                   num_llr_bins, /*is_last_iteration_node_update=*/false);
    // Now we have check_out_density, and we can propagate this through a random
    // node.
    bintree_update(dist.node, check_out_density, node_out_density, prod_cache,
                   num_llr_bins,
                   /*is_last_iteration_node_update=*/(t + 1 == num_iterations));
    // Must take the product with the initial belief from the channel parameters
    fast_op(node_out_density, initial_density, tmp, prod_cache, num_llr_bins);
    std::swap(tmp, node_out_density);
    zfill(tmp);

    // We have now prepared node_out_density for the this iteration so we can
    // replace the current density with it.
    std::swap(density, node_out_density);
    // We should zfill node_out_density so we can reuse it later
    zfill(node_out_density);
    density_callback(density);

    // We are finished using check_out_density so we zfill it
    zfill(check_out_density);

    if (!early_stop) continue;

    // Check the early stopping conditions
    double error_rate = 0;
    for (size_t i = ((num_llr_bins + 1) >> 1); i < num_llr_bins; ++i) {
      assert(quantized_llrs[i] >= 0);
      error_rate += density[i];
    }
    bool below = error_rate < (delta * shrink_factor);
    if (below) break;
    if (error_rate > delta * 1.1) break;
  }
  double error_rate = 0;
  for (size_t i = ((num_llr_bins + 1) >> 1); i < num_llr_bins; ++i) {
    assert(quantized_llrs[i] >= 0);
    error_rate += density[i];
  }
  bool below = error_rate < (delta * shrink_factor);
  if (verbose) {
    std::cout << "final error_rate = " << error_rate << "\n";
  }
  return below;
}

std::pair<double, double> binary_search_threshold(
    const DegreeDistributionPair& dist, double delta_min, double delta_max,
    double precision, size_t num_llr_bins, size_t num_iterations,
    bool fractional_quantization, bool early_stop, bool quantization_fiddling,
    double shrink_factor, bool verbose, double halt_if_leq) {
  while (delta_max - delta_min > delta_min * precision) {
    double delta = (delta_max + delta_min) / 2;
    if (verbose) {
      std::cout << "trying with delta = " << delta << "\n";
    }
    bool below = density_evolution(
        dist, delta, num_llr_bins, num_iterations, fractional_quantization,
        early_stop, quantization_fiddling, shrink_factor, verbose);
    if (below) {
      delta_min = delta;
    } else {
      delta_max = delta;
    }
    if (verbose) {
      std::cout << "got below = " << below << " new delta range " << delta_min
                << " - " << delta_max << "\n";
    }
    if (halt_if_leq > 0 and delta_max <= halt_if_leq) {
      break;
    }
  }
  return {delta_min, delta_max};
}
