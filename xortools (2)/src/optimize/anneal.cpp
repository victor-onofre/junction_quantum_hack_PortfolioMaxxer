#include "anneal.hpp"

#include <cassert>
#include <set>

constexpr double ACCEPTANCE_THRESHOLD_FOR_INCREMENTAL_TRACKING = 0.03;

int evaluate(const sparse::SparseMatrix& H, const std::vector<int>& target,
             const std::vector<bool>& x) {
  assert(H.ncols == target.size() && H.nrows == x.size());
  int num_clause_sat = 0;
  for (size_t j = 0; j < H.ncols; ++j) {
    int yj = 0;
    for (const auto& it : H.col_entries[j]) {
      yj ^= x[it.first];
    }
    num_clause_sat += (yj == target[j]);
  }
  return num_clause_sat;
}

int difference(const sparse::SparseMatrix& H, std::vector<bool>& y, size_t i) {
  int diff = 0;
  for (const size_t& j : H.row_entries_vec[i]) {
    diff += y[j];
  }
  diff *= 2;
  diff -= H.row_entries_vec[i].size();
  return diff;
}

inline int difference_incremental(std::vector<int>& diffs, size_t i) {
  return diffs[i];
}

std::vector<bool> anneal(const sparse::SparseMatrix& H,
                         const std::vector<int>& target, size_t iterations,
                         double min_beta, double max_beta,
                         double min_acceptance_rate,
                         std::function<uint64_t(void)> gen, bool verbose) {
  std::vector<bool> x;
  for (size_t i = 0; i < H.nrows; ++i) {
    x.push_back(gen() & 1);
  }
  return anneal(H, target, x, iterations, min_beta, max_beta,
                min_acceptance_rate, gen, verbose);
}

// Fairly well-optimized implementation of plain vanilla simulated annealing
std::vector<bool> anneal(const sparse::SparseMatrix& H,
                         const std::vector<int>& target, std::vector<bool>& x,
                         size_t iterations, double min_beta, double max_beta,
                         double min_acceptance_rate,
                         std::function<uint64_t(void)> gen, bool verbose) {
  if (x.size() != H.nrows) {
    throw std::invalid_argument("Wrong size for x. Got " +
                                std::to_string(x.size()) + ". Expected " +
                                std::to_string(H.nrows) + ".");
  }
  double delta_beta = (max_beta - min_beta) / double(iterations);
  assert(delta_beta > 0 && "would take infinite time");

  if (verbose) {
    std::cout << "H.ncols = " << H.ncols << " H.nrows = " << H.nrows
              << " target.size() = " << target.size() << "\n";
  }
  std::vector<bool> y(H.ncols, false);
  for (size_t i = 0; i < H.nrows; ++i) {
    for (const size_t& j : H.row_entries_vec[i]) {
      y[j] = (y[j] != bool(x[i]));
    }
  }
  for (size_t j = 0; j < H.ncols; ++j) {
    y[j] = (y[j] != bool(target[j]));
  }
  int current_val = evaluate(H, target, x);
  if (verbose) {
    std::cout << "initial val = " << current_val << std::endl;
  }
  std::vector<int> diffs(H.nrows);
  int best_val = current_val;
  std::vector<bool> best_x = x;
  size_t count = 0;
  bool incremental_tracking = false;
  size_t print_freq = std::max(1000ul, iterations >> 9);
  for (double beta = min_beta; beta <= max_beta; beta += delta_beta) {
    double acceptance_rate = 0;
    for (size_t i = 0; i < H.nrows; ++i) {
      // Consider flipping i
      int diff;
      // There are two ways of computing the diffs. The first way is on-the-fly
      // and the second way is incremental. Incremental tracking does extra work
      // whenever there is an update to keep the diffs array up-to-date.
      // On-the-fly tends to be the fastest option when beta is small and we do
      // lots of updates. Incremental tends to be much faster when beta is large
      // and we do few updates, since the extra work in the diffs array happens
      // rarely and most of the time we just sample a random value and move on
      // to the next i. We switch between the two modes dynamically to get the
      // best of both worlds.
      if (incremental_tracking) {
        diff = difference_incremental(diffs, i);
      } else {
        diff = difference(H, y, i);
      }
      // Note that if the new val is lower than the current val, the probability
      // is smaller.
      double p_accept = std::exp(beta * double(diff));
      // Sample from the random number generator to see if we accept it
      if (double(gen()) / double(UINT64_MAX) < p_accept) {
        // It is accepted. Update x, the current_val and best_val
        x[i] = !x[i];
        for (const size_t& j : H.row_entries_vec[i]) {
          y[j] = !y[j];

          if (!incremental_tracking) continue;

          for (const size_t& i1 : H.col_entries_vec[j]) {
            diffs[i1] -= 2 - 4 * y[j];
          }
        }
        current_val += diff;
        if (current_val > best_val) {
          best_val = current_val;
          best_x = x;
        }
        acceptance_rate += 1;
      }
    }
    acceptance_rate /= double(H.nrows);
    if (!incremental_tracking and
        acceptance_rate < ACCEPTANCE_THRESHOLD_FOR_INCREMENTAL_TRACKING) {
      if (verbose) {
        std::cout << "activating incremental tracking mode since changes are "
                     "infrequent\n";
      }
      incremental_tracking = true;
      for (size_t i = 0; i < H.nrows; ++i) {
        diffs[i] = difference(H, y, i);
      }
    }
    // assert(evaluate(H, target, x)==current_val);
    if (count++ % print_freq == 0 and verbose) {
      std::cout << "beta = " << beta << " current_val = " << current_val
                << " best_val = " << best_val
                << " acceptance_rate = " << acceptance_rate << std::endl;
    }
    if (acceptance_rate < min_acceptance_rate) {
      if (verbose) {
        std::cout << "ending early due to small acceptance_rate\n";
      }
      break;
    }
  }
  return best_x;
}

// Discount factor used to downweight high-degree clauses at low values of beta.
// Heuristic function.
double discount_factor(double beta, double degree) {
  return 1.0 - std::min(1.0, std::exp(-beta / degree));
}

// Discounted objective function that reduces the impact of high-degree clauses
// at low values of beta.
double discounted_evaluate(double beta, const sparse::SparseMatrix& H,
                           const std::vector<int>& target,
                           const std::vector<bool>& x) {
  assert(H.ncols == target.size() && H.nrows == x.size());
  double total = 0;
  for (size_t j = 0; j < H.ncols; ++j) {
    int yj = 0;
    for (const auto& it : H.col_entries[j]) {
      yj ^= x[it.first];
    }
    if (yj == target[j]) {
      total += discount_factor(beta,
                               /*degree=*/H.col_entries_vec[j].size()
                               // /*degree=*/double(j)/double(H.ncols) * 10
      );
    }
  }
  return total;
}

// Difference function for the discounted objective function
double discounted_difference(const double beta, const sparse::SparseMatrix& H,
                             std::vector<bool>& y, size_t i) {
  double total = 0;
  double diff = 0;
  for (const size_t& j : H.row_entries_vec[i]) {
    double factor = discount_factor(beta,
                                    /*degree=*/H.col_entries_vec[j].size()
                                    // /*degree=*/double(j)/double(H.ncols) * 10
    );
    total += factor;
    diff += y[j] * factor;
  }
  diff *= 2;
  diff -= total;
  return diff;
}

// Modified simulated annealing algorithm where we optimize a dynamic objective
// function that discounts high-degree clauses at lower values of beta.
std::vector<bool> irreg_anneal(const sparse::SparseMatrix& H,
                               const std::vector<int>& target,
                               size_t iterations,
                               std::function<uint64_t(void)> gen,
                               bool verbose) {
  std::vector<bool> x;
  for (size_t i = 0; i < H.nrows; ++i) {
    x.push_back(gen() & 1);
  }

  if (x.size() != H.nrows) {
    throw std::invalid_argument("Wrong size for x. Got " +
                                std::to_string(x.size()) + ". Expected " +
                                std::to_string(H.nrows) + ".");
  }

  double max_clause_degree = 0;
  for (size_t j = 0; j < H.ncols; ++j) {
    max_clause_degree =
        std::max(max_clause_degree, double(H.col_entries_vec[j].size()));
  }
  double max_beta = -max_clause_degree * std::log(0.1);
  if (verbose) {
    std::cout << "max clause degree = " << max_clause_degree
              << "; chose max_beta = " << max_beta << "\n";
  }
  double min_beta = 0;
  double delta_beta = (max_beta - min_beta) / double(iterations);
  assert(delta_beta > 0 && "would take infinite time");

  if (verbose) {
    std::cout << "H.ncols = " << H.ncols << " H.nrows = " << H.nrows
              << " target.size() = " << target.size() << "\n";
  }
  std::vector<bool> y(H.ncols, false);
  for (size_t i = 0; i < H.nrows; ++i) {
    for (const size_t& j : H.row_entries_vec[i]) {
      y[j] = (y[j] != bool(x[i]));
    }
  }
  for (size_t j = 0; j < H.ncols; ++j) {
    y[j] = (y[j] != bool(target[j]));
  }

  std::vector<bool> best_x = x;
  double beta = min_beta;
  int current_num_sat = evaluate(H, target, x);
  int best_num_sat = current_num_sat;
  size_t count = 0;
  size_t print_freq = std::max(1000ul, iterations >> 9);
  for (; beta <= max_beta; beta += delta_beta) {
    double acceptance_rate = 0;
    for (size_t i = 0; i < H.nrows; ++i) {
      // Consider flipping i
      double diff = discounted_difference(beta, H, y, i);
      int diff_num_sat = difference(H, y, i);
      // Note that if the diff is negative, the probability
      // is smaller.
      double p_accept = std::exp(beta * double(diff));
      // Sample from the random number generator to see if we accept it
      if (double(gen()) / double(UINT64_MAX) < p_accept) {
        // It is accepted. Update x, the current_num_sat and best_num_sat
        x[i] = !x[i];
        for (const size_t& j : H.row_entries_vec[i]) {
          y[j] = !y[j];
        }
        current_num_sat += diff_num_sat;
        if (current_num_sat > best_num_sat) {
          best_num_sat = current_num_sat;
          best_x = x;
        }
        acceptance_rate += 1;
      }
    }
    acceptance_rate /= double(H.nrows);
    if (acceptance_rate < 0.01) {
      delta_beta = (max_beta - min_beta) / double(iterations) /
                   std::max(acceptance_rate, 0.00001);
    }
    // assert(evaluate(H, target, x)==current_num_sat);
    if (count++ % print_freq == 0) {
      if (verbose) {
        std::cout << "beta = " << beta << " delta_beta = " << delta_beta
                  << " current_num_sat = " << current_num_sat
                  << " best_num_sat = " << best_num_sat
                  << " acceptance_rate = " << acceptance_rate << "\n";
      }
    }
  }
  return best_x;
}
