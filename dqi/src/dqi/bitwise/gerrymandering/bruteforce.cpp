
#include <assert.h>

#include <numeric>
#include <sstream>
#include <stdexcept>

#include "problem_def.h"

namespace {
MaximizationResult _backtrack(
    int n, int s, int budget, const probability_distribution_t& probability,
    const probability_distribution_t& probability_for_cost, const int& target) {
  s = std::min(s, budget);
  if (n * (long long)s <= budget) {
    auto new_probability = convolve(probability, n, probability_for_cost[s]);
    auto res = MaximizationResult::constant(n, 0, s);
    res.phi_c = std::accumulate(new_probability.begin() + target,
                                new_probability.end(), (probability_t)0);
    return res;
  }
  if (n == 0) {
    if (probability.size() <= target) return MaximizationResult::constant(0);
    auto res = MaximizationResult::constant(0);
    res.phi_c = std::accumulate(probability.begin() + target, probability.end(),
                                (probability_t)0);
    return res;
  }
  auto best_prob =
      _backtrack(n, s - 1, budget, probability, probability_for_cost, target);

  auto p = probability_for_cost[s];
  auto new_prob = convolve(probability, p);
  auto tmp =
      _backtrack(n - 1, s, budget - s, new_prob, probability_for_cost, target);
  tmp.choices.push_back(s);
  if (best_prob < tmp) {
    best_prob.swap(tmp);
  }
  return best_prob;
}
}  // namespace

namespace bruteforce {
MaximizationResult maximize_probability(const GerrymanderingInstance& gi) {
  if (gi.target > gi.m) {
    return MaximizationResult::constant(gi.m);
  }
  probability_distribution_t probability{1};
  return _backtrack(gi.m, gi.b, gi.budget, probability, gi.p_s, gi.target);
}
};  // namespace bruteforce
