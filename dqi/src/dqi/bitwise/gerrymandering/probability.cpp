#include "probability.h"

#include <math.h>

log_probability_t add_logs(log_probability_t log_a, log_probability_t log_b) {
  if (log_a > log_b) std::swap(log_a, log_b);
  return log_b + std::log(1 + std::exp(log_a - log_b));
}

void normalize(probability_distribution_t & d) {
  probability_t s = 0;
  for(auto p : d) s += p;
  for(auto & p : d) p /= s;
}

probability_distribution_t convolve(const probability_distribution_t& pt,
                                    probability_t p) {
  probability_distribution_t new_pt(pt.size() + 1);
  for (int i = 0; i < (int)new_pt.size(); i++) {
    new_pt[i] =
        ((i < pt.size()) ? pt[i] : 0) * (1 - p) + p * (i ? pt[i - 1] : 0);
  }
  normalize(new_pt);
  return new_pt;
}

probability_distribution_t binomial(int n, probability_t p) {
  const long double log_p = std::log(std::max(p, (probability_t)1e-100));
  const long double log_one_m_p =
      std::log(std::max(1 - p, (probability_t)1e-100));

  std::vector<long double> log_factorial(n + 1, 0);
  for (int i = 1; i <= n; i++) {
    log_factorial[i] = log_factorial[i - 1] + std::log(i);
  }

  probability_distribution_t ret(n + 1);
  for (int i = 0; i <= n; i++) {
    auto log_q = log_factorial[n] - log_factorial[i] - log_factorial[n - i] +
                 i * log_p + (n - i) * log_one_m_p;
    ret[i] = std::exp(log_q);
  }
  return ret;
}

probability_distribution_t convolve(const probability_distribution_t& u,
                                    const probability_distribution_t& v) {
  probability_distribution_t ret(u.size() + v.size() - 1);
  for (int i = 0; i < (int)u.size(); i++) {
    for (int j = 0; j < (int)v.size(); j++) {
      ret[i + j] += u[i] * v[j];
    }
  }

  normalize(ret);
  return ret;
}

probability_distribution_t convolve(const probability_distribution_t& pt, int n,
                                    probability_t p) {
  auto bin = binomial(n, p);
  return convolve(pt, bin);
}

probability_distribution_t distribution_from_probabilities(
    const std::vector<probability_t>& probs) {
  probability_distribution_t res{1.0};
  for (const auto& p : probs) {
    res.push_back(0);
    for (int i = (int)res.size() - 1; i >= 0; i--) {
      res[i] = (1 - p) * res[i] + (i ? p * res[i - 1] : 0);
    }
  }
  normalize(res);
  return res;
}

log_probability_distribution_t log_distribution_from_probabilities(
    const std::vector<probability_t>& probs) {
  log_probability_distribution_t res{0};
  for (const auto& p : probs) {
    auto log_p = std::log(p);
    auto log_one_m_p = std::log(std::max(1 - p, (probability_t)-100));
    res.push_back(log_p + res.back());
    for (int i = (int)res.size() - 2; i >= 0; i--) {
      res[i] = log_one_m_p + res[i];
      if (i) {
        res[i] = add_logs(res[i], log_p + res[i - 1]);
      }
    }
  }
  return res;
}
