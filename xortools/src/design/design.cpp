#include "design.hpp"

#include <cassert>
#include <thread>

#include "belief/decoding.hpp"
#include "belief/density.hpp"
#include "irreg/irreg.hpp"
#include "optimize/anneal.hpp"
#include "optimize/problem.hpp"

double est_dqival(double delta) {
  return 0.5 + std::sqrt(0.25 - (0.5 - delta) * (0.5 - delta));
}

double est_thval(double dual_code_rate) { return 1 - dual_code_rate / 2.0; }

double est_thval(const DegreeDistributionPair& dist) {
  return est_thval(dist.rate());
}

double est_saval(double D) {
  return std::min(1.0, 0.5 + std::exp(0.82) * std::pow(D, -0.44547) / 2.0);
}

double est_adaptive_saval_model(DegreeDistributionPair dist) {
  double rate = dist.rate();
  // Considers several possible preprocessing steps combined with simulated
  // annealing
  dist = dist.vertex_perspective();
  double best_val = 0;
  // Suppose we ignore the nodes of degree at least d.
  double cum_prob = 0;
  double cum_sum_degs = 0;
  for (size_t i = 0; i < dist.node.degrees.size(); ++i) {
    cum_prob += dist.node.probabilities[i];
    cum_sum_degs += dist.node.probabilities[i] * dist.node.degrees[i];
    double avg_node_degree = cum_sum_degs / cum_prob;
    double avg_check_degree = avg_node_degree / (1 - rate) * cum_prob;
    double trunc_val =
        cum_prob * est_saval(avg_check_degree) + (1 - cum_prob) / 2.0;
    best_val = std::max(best_val, trunc_val);
  }
  return best_val;
}

double est_saval_empirical(DegreeDistributionPair dist, size_t num_clauses,
                           size_t num_samples, size_t num_iterations,
                           uint64_t seed, std::vector<double>& values) {
  constexpr double kMinBeta = 0;
  constexpr double kMaxBeta = 10;
  constexpr double kMinAcceptanceRate = 0.000001;
  constexpr bool verbose = false;

  values.resize(num_samples);
  std::vector<std::thread> threads;
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<uint64_t> uniform;
  for (size_t t = 0; t < num_samples; ++t) {
    uint64_t thread_seed = uniform(eng);
    threads.push_back(std::thread(
        [&values, thread_seed, t, &dist, num_iterations, num_clauses]() {
          std::mt19937_64 eng(thread_seed);
          std::uniform_int_distribution<uint64_t> uniform;
          auto gen = [&]() { return uniform(eng); };

          size_t n = std::max(
              1ul, size_t(std::round(double(num_clauses) * (1 - dist.rate()))));
          sparse::SparseMatrix H =
              sample_irreg(num_clauses, n, dist.vertex_perspective(), gen,
                           /*verbose=*/false);
          std::vector<int> target(num_clauses);
          for (size_t i = 0; i < num_clauses; ++i) {
            if (gen() % 2) {
              target[i] = 1;
            }
          }
          std::vector<bool> x = anneal(H, target, num_iterations, kMinBeta,
                                       kMaxBeta, kMinAcceptanceRate, gen,
                                       /*verbose=*/false);
          int num_sat_full = evaluate(H, target, x);

          values[t] = double(num_sat_full) / double(num_clauses);
        }));
  }
  for (size_t t = 0; t < num_samples; ++t) {
    threads[t].join();
  }
  std::sort(values.begin(), values.end());
  // double median = values[(values.size() - 1) >> 1];
  // return median;
  double mean = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    mean += values[i];
  }
  mean /= double(values.size());
  return mean;
}

double est_stripped_saval_empirical(DegreeDistributionPair dist,
                                    size_t num_clauses, size_t num_samples,
                                    size_t num_iterations, uint64_t seed) {
  constexpr bool verbose = false;

  std::vector<double> vals(num_samples);
  std::vector<std::thread> threads;
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<uint64_t> uniform;
  for (size_t t = 0; t < num_samples; ++t) {
    uint64_t thread_seed = uniform(eng);
    threads.push_back(std::thread([&vals, thread_seed, t, &dist, num_iterations,
                                   num_clauses]() {
      std::mt19937_64 eng(thread_seed);
      std::uniform_int_distribution<uint64_t> uniform;
      auto gen = [&]() { return uniform(eng); };

      size_t n = std::max(
          1ul, size_t(std::round(double(num_clauses) * (1 - dist.rate()))));
      sparse::SparseMatrix H =
          sample_irreg(num_clauses, n, dist.vertex_perspective(), gen,
                       /*verbose=*/false);
      std::vector<int> target(num_clauses);
      for (size_t i = 0; i < num_clauses; ++i) {
        if (gen() % 2) {
          target[i] = 1;
        }
      }

      std::map<int, MaxXORSATProblem> subproblems;
      MaxXORSATProblem problem{H, target};
      strip_high_degree_clauses(problem, subproblems, /*verbose=*/false);

      std::vector<std::thread> threads;
      size_t num_subproblems = subproblems.size();
      std::vector<int> subproblem_num_sat(num_subproblems);
      size_t counter = 0;
      for (const auto& [degree, subproblem] : subproblems) {
        uint64_t subthread_seed = gen();
        threads.push_back(
            std::thread([&subproblem_num_sat, &subproblems, &H, &target, degree,
                         num_iterations, subthread_seed, counter]() {
              constexpr double kMinBeta = 0;
              constexpr double kMaxBeta = 5;
              // constexpr double kMinAcceptanceRate = 0.00001;
              constexpr double kMinAcceptanceRate = 0;

              std::mt19937_64 eng(subthread_seed);
              std::uniform_int_distribution<uint64_t> uniform;
              auto gen = [&]() -> uint64_t { return uniform(eng); };
              auto x = anneal(subproblems[degree].H,
                              subproblems[degree].targets, num_iterations,
                              kMinBeta, kMaxBeta, kMinAcceptanceRate, gen,
                              /*verbose=*/false);
              int num_sat_full = evaluate(H, target, x);
              if (verbose) {
                std::cout << "stripped clauses of degree > " << degree
                          << " annealed solution satisfying " << num_sat_full
                          << " clauses on the full problem\n";
              }
              subproblem_num_sat[counter] = num_sat_full;
            }));
        ++counter;
      }
      assert(threads.size() == num_subproblems);
      assert(subproblem_num_sat.size() == num_subproblems);
      int best_num_sat_full = 0;
      for (size_t i = 0; i < num_subproblems; ++i) {
        threads[i].join();
        best_num_sat_full = std::max(best_num_sat_full, subproblem_num_sat[i]);
      }
      vals[t] = double(best_num_sat_full) / double(num_clauses);
    }));
  }
  for (size_t t = 0; t < num_samples; ++t) {
    threads[t].join();
  }
  double sum = 0;
  for (size_t i = 0; i < vals.size(); ++i) {
    sum += vals[i];
  }
  return sum / double(vals.size());
}

static std::mutex io_mutex;

template <typename T>
std::vector<T> apply_sort(const std::vector<T>& v,
                          const std::vector<size_t>& idx) {
  assert(v.size() == idx.size());
  std::vector<T> w(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    w[i] = v[idx[i]];
  }
  return w;
}

template <typename T>
void print_vec(std::ostream& out, const std::vector<T>& v) {
  out << "{";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) out << ", ";
    out << v[i];
  }
  out << "}";
}

double est_quantum_advantage_over_sa_empirical(
    DegreeDistributionPair dist, size_t num_clauses, size_t num_samples,
    size_t num_iterations, double max_error_rate, int max_iter,
    size_t shots_per_point, uint64_t seed) {
  // constexpr size_t kInstanceNumClauses = 5000;
  // constexpr size_t kNumSamples = 8;
  // constexpr size_t kNumAnnealingIterations = 1000;

  constexpr double kMinBeta = 0;
  constexpr double kMaxBeta = 5;
  constexpr double kMinAcceptanceRate = 0;
  constexpr bool verbose = false;

  // constexpr size_t kMaxIter = 50;
  // constexpr double kMaxErrorRate = 0.01;
  // constexpr double kShotsPerPoint = 200;

  std::vector<double> th_vals(num_samples);
  std::vector<double> annealing_vals(num_samples);
  std::vector<double> dqi_vals(num_samples);
  std::vector<double> delta_vals(num_samples);
  std::vector<double> ratio_vals(num_samples);
  std::vector<std::thread> threads;
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<uint64_t> uniform;
  size_t threads_per_sample =
      std::max(1ul, std::thread::hardware_concurrency() / num_samples);
  auto [lo, hi] = binary_search_threshold(
      dist, 0, 0.5, /*precision=*/0.0001, 151, 100,
      /*fractional_quantization=*/false, /*early_stop=*/true,
      /*quantization_fiddling=*/false, /*shrink_factor=*/0.0001, verbose,
      /*halt_if_leq=*/0);
  double de_delta = lo;
  for (size_t t = 0; t < num_samples; ++t) {
    uint64_t thread_seed = uniform(eng);
    threads.push_back(std::thread([&th_vals, &annealing_vals, &dqi_vals,
                                   &delta_vals, &ratio_vals, thread_seed, t,
                                   &dist, num_clauses, &threads_per_sample,
                                   num_iterations, max_error_rate, max_iter,
                                   shots_per_point, &de_delta]() {
      std::mt19937_64 eng(thread_seed);
      std::uniform_int_distribution<uint64_t> uniform;
      auto gen = [&]() -> uint64_t { return uniform(eng); };

      size_t n = std::max(
          1ul, size_t(std::round(double(num_clauses) * (1 - dist.rate()))));
      sparse::SparseMatrix H =
          sample_irreg(num_clauses, n, dist.vertex_perspective(), gen,
                       /*verbose=*/false);

      std::vector<int> target(num_clauses);
      for (size_t i = 0; i < num_clauses; ++i) {
        if (gen() % 2) {
          target[i] = 1;
        }
      }

      MaxXORSATProblem problem;
      problem.H = H;
      problem.targets = target;

      int best_num_sat_full;
      std::thread annealing_thread([&]() {
        // std::map<int, MaxXORSATProblem> subproblems;
        // strip_high_degree_clauses(problem, subproblems,
        // /*verbose=*/false);

        std::vector<bool> x =
            anneal(problem.H, problem.targets, num_iterations, kMinBeta,
                   kMaxBeta, kMinAcceptanceRate, gen, /*verbose=*/false);
        best_num_sat_full = evaluate(H, target, x);
      });
      double delta;
      // delta = get_max_tolerable_error_rate(
      //       problem.H, max_error_rate, shots_per_point, max_iter,
      //       /*num_threads=*/threads_per_sample, eng,
      //       /*verbose=*/false);
      delta = de_delta;
      double quantum_value = est_dqival(delta);
      annealing_thread.join();
      double annealing_value = double(best_num_sat_full) / double(num_clauses);

      double th_value = est_thval(1 - double(H.nrows) / double(H.ncols));
      double classical_value = std::max(annealing_value, th_value);

      th_vals[t] = th_value;
      annealing_vals[t] = annealing_value;
      dqi_vals[t] = quantum_value;
      delta_vals[t] = delta;
      ratio_vals[t] = quantum_value / classical_value;

      if (verbose) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cout << "thread " << t << " decoding delta = " << delta
                  << " DQI value = " << quantum_value
                  << " annealing_value = " << annealing_value
                  << " th_value = " << th_value
                  << " classical_value = " << classical_value
                  << " advantage_ratio = " << ratio_vals[t] << std::endl;
      }
    }));
  }
  for (size_t t = 0; t < num_samples; ++t) {
    threads[t].join();
  }
  std::vector<size_t> idx = argsort(ratio_vals);
  th_vals = apply_sort(th_vals, idx);
  annealing_vals = apply_sort(annealing_vals, idx);
  delta_vals = apply_sort(delta_vals, idx);
  dqi_vals = apply_sort(dqi_vals, idx);
  ratio_vals = apply_sort(ratio_vals, idx);

  double sum = 0;
  double num = 0;
  for (size_t i = 0; i < ratio_vals.size(); ++i) {
    sum += ratio_vals[i];
    num += 1;
  }
  double mean = sum / num;
  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << "dist = ";
    std::cout << dist.degfile_str() << std::endl;
    std::cout << "th_vals = ";
    print_vec(std::cout, th_vals);
    std::cout << std::endl;
    std::cout << "annealing_vals = ";
    print_vec(std::cout, annealing_vals);
    std::cout << std::endl;
    std::cout << "delta_vals = ";
    print_vec(std::cout, delta_vals);
    std::cout << std::endl;
    std::cout << "dqi_vals = ";
    print_vec(std::cout, dqi_vals);
    std::cout << std::endl;
    std::cout << "ratio_values = ";
    print_vec(std::cout, ratio_vals);
    std::cout << std::endl;
    std::cout << "mean = " << mean << std::endl;
  }
  return mean;
}

double est_adaptive_saval_empirical(DegreeDistributionPair dist,
                                    uint64_t seed) {
  constexpr size_t kInstanceNumClauses = 5000;
  constexpr size_t kNumSamples = 5;
  constexpr size_t kNumAnnealingIterations = 250'000;

  std::vector<double> vals(kNumSamples);
  std::vector<std::thread> threads;
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<uint64_t> uniform;
  for (size_t t = 0; t < kNumSamples; ++t) {
    uint64_t thread_seed = uniform(eng);
    threads.push_back(std::thread([&vals, thread_seed, t, &dist,
                                   kNumAnnealingIterations]() {
      std::mt19937_64 eng(thread_seed);
      std::uniform_int_distribution<uint64_t> uniform;
      auto gen = [&]() { return uniform(eng); };

      size_t n = std::max(1ul, size_t(std::round(double(kInstanceNumClauses) *
                                                 (1 - dist.rate()))));
      sparse::SparseMatrix H =
          sample_irreg(kInstanceNumClauses, n, dist.vertex_perspective(), gen,
                       /*verbose=*/false);
      std::vector<int> target(kInstanceNumClauses);
      for (size_t i = 0; i < kInstanceNumClauses; ++i) {
        if (gen() % 2) {
          target[i] = 1;
        }
      }
      std::vector<bool> x =
          irreg_anneal(H, target, /*iterations=*/kNumAnnealingIterations, gen,
                       /*verbose=*/false);
      int num_sat = evaluate(H, target, x);
      vals[t] = double(num_sat) / double(kInstanceNumClauses);
    }));
  }
  for (size_t t = 0; t < kNumSamples; ++t) {
    threads[t].join();
  }
  std::sort(vals.begin(), vals.end());
  double median = vals[(vals.size() - 1) >> 1];
  return median;
}

double est_decoding_threshold_empirical(DegreeDistributionPair dist,
                                        size_t num_nodes, size_t num_samples,
                                        double max_error_rate, int max_iter,
                                        size_t shots_per_point, uint64_t seed) {
  std::vector<double> vals(num_samples);
  std::vector<std::thread> threads;
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<uint64_t> uniform;
  size_t threads_per_sample =
      std::max(1ul, std::thread::hardware_concurrency() / num_samples);
  for (size_t t = 0; t < num_samples; ++t) {
    uint64_t thread_seed = uniform(eng);
    threads.push_back(std::thread([&vals, thread_seed, t, &dist, num_nodes,
                                   &threads_per_sample, max_iter,
                                   max_error_rate, shots_per_point]() {
      std::mt19937_64 eng(thread_seed);
      std::uniform_int_distribution<uint64_t> uniform;
      auto gen = [&]() { return uniform(eng); };

      size_t n = std::max(
          1ul, size_t(std::round(double(num_nodes) * (1 - dist.rate()))));
      sparse::SparseMatrix H =
          sample_irreg(num_nodes, n, dist.vertex_perspective(), gen,
                       /*verbose=*/false);

      vals[t] = get_max_tolerable_error_rate(H, max_error_rate, shots_per_point,
                                             /*threads=*/threads_per_sample,
                                             max_iter, eng, /*verbose=*/false);
    }));
  }
  for (size_t t = 0; t < num_samples; ++t) {
    threads[t].join();
  }
  double sum = 0;
  for (size_t i = 0; i < num_samples; ++i) {
    sum += vals[i];
  }
  return sum / double(num_samples);
}
