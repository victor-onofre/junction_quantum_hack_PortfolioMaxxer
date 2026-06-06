#ifndef _GERRYMANDERING_KNAPSACK_H
#define _GERRYMANDERING_KNAPSACK_H

#include <numeric>

#include "problem_def.h"

namespace knapsack {
struct Result {
  probability_distribution_t probs;
  std::vector<int> choices;

  static Result EmptyResult();

  probability_t ccdf(int) const;
  probability_t potential(int, probability_t) const;

  bool is_better_than(const Result&, int,
                      const probability_distribution_t&) const;

  MaximizationResult to_maximization_result(int) const;

  Result convolve(int s, probability_t p) const;

  friend std::ostream& operator<<(std::ostream& os, const Result& r) {
    os << "Result{" << r.probs << " | " << r.choices << "}";
    return os;
  }

  void clear();

  bool is_valid(int) const;
};

struct ResultCollection {
  std::vector<Result> results;

  ResultCollection convolve(int, probability_t);
  void merge(const ResultCollection& other, int, int);
  void clear();
};
MaximizationResult maximize_probability(const GerrymanderingInstance&, bool);
};  // namespace knapsack

#endif
