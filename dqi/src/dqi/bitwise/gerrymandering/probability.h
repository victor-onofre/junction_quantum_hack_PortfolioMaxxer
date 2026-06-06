#ifndef _PROBABILITY_H
#define _PROBABILITY_H

#include <math.h>

#include <ostream>
#include <vector>

using probability_t = long double;
using log_probability_t = long double;
using probability_distribution_t = std::vector<probability_t>;
using log_probability_distribution_t = std::vector<log_probability_t>;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  bool first_item = true;
  for (auto x : vec) {
    if (!first_item) {
      os << ", ";
    }
    first_item = false;
    os << x;
  }
  os << "]";
  return os;
}

log_probability_t add_logs(log_probability_t, log_probability_t);

probability_distribution_t binomial(int, probability_t);

probability_distribution_t convolve(const probability_distribution_t& pt,
                                    probability_t p);
probability_distribution_t convolve(const probability_distribution_t& pt, int n,
                                    probability_t p);
probability_distribution_t convolve(const probability_distribution_t&,
                                    const probability_distribution_t&);

probability_distribution_t distribution_from_probabilities(
    const std::vector<probability_t>&);
log_probability_distribution_t log_distribution_from_probabilities(
    const std::vector<probability_t>&);

#endif
