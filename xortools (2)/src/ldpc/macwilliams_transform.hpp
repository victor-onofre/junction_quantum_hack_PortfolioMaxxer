#ifndef __MW_TF_HPP__
#define __MW_TF_HPP__
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "../lattice/gmpwrap.hpp"

// Function to load weight distribution from a file into a vector of GmpRational
void load_weight_dist(const std::string& filename,
                      std::vector<gmpwrap::GmpRational>& code_weight_dist) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Error: Unable to open file " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    try {
      gmpwrap::GmpRational value;
      assert(mpq_set_str(value.value, line.c_str(), 10) == 0);
      code_weight_dist.push_back(value);
    } catch (const std::exception& e) {
      std::cerr << "Error parsing line: " << line << "\n"
                << e.what() << std::endl;
      throw e;
    }
  }
  file.close();
}

gmpwrap::GmpRational binom(size_t n, size_t k) {
  // Use GMP's mpz_bin_uiui to calculate the binomial coefficient
  mpz_class binomial_value;
  mpz_bin_uiui(binomial_value.get_mpz_t(), n, k);

  // Create a GmpRational representing an integer
  gmpwrap::GmpRational result;
  mpq_set_num(result.value, binomial_value.get_mpz_t());
  mpq_set_den(result.value, mpz_class(1).get_mpz_t());  // Denominator = 1
  return result;
}

void init_binom_cache(
    std::vector<std::vector<gmpwrap::GmpRational>>& binom_cache, size_t max_n,
    size_t num_worker_threads) {
  // Resize binom_cache in the main thread
  binom_cache.resize(max_n + 1);
  for (size_t n = 0; n <= max_n; ++n) {
    binom_cache[n].resize(n + 1);
  }

  // Prepare a vector of all (n, k) pairs
  std::vector<std::pair<size_t, size_t>> tasks;
  for (size_t n = 0; n <= max_n; ++n) {
    for (size_t k = 0; k <= n; ++k) {
      tasks.emplace_back(n, k);
    }
  }

  // Atomic counter for the next unclaimed task
  std::atomic<size_t> next_unclaimed_task{0};

  // Vector of threads
  std::vector<std::thread> worker_threads;

  // Spawn worker threads
  for (size_t t = 0; t < num_worker_threads; ++t) {
    worker_threads.emplace_back([&]() {
      size_t task_index;
      while ((task_index = next_unclaimed_task++) < tasks.size()) {
        const auto& [n, k] = tasks[task_index];
        binom_cache[n][k] = binom(n, k);
      }
    });
  }

  // Join all threads
  for (auto& thread : worker_threads) {
    thread.join();
  }
}

gmpwrap::GmpRational krawtchouk(
    const size_t w, const size_t t, const size_t m,
    const std::vector<std::vector<gmpwrap::GmpRational>>& binom_cache) {
  gmpwrap::GmpRational total = 0;
  gmpwrap::GmpRational sign = 1;
  // for (size_t j = (w + t > m ? ((w+t)-m):0); j <= std::min(w, t); ++j) {
  size_t start = 0;
  if (w + t > m) {
    start = (w + t) - m;
    if (start % 2) {
      sign = -1;
    }
  }
  for (size_t j = start; j <= std::min(w, t); ++j) {
    total += sign * binom_cache[t][j] * binom_cache[m - t][w - j];
    sign *= -1;
  }
  return total;
}

void macwilliams_transform(std::vector<gmpwrap::GmpRational>& code,
                           std::vector<gmpwrap::GmpRational>& dual_code,
                           size_t num_worker_threads) {
  size_t m = code.size() - 1;

  // Calculate code size
  gmpwrap::GmpRational code_size(0);
  for (size_t i = 0; i <= m; ++i) {
    code_size += code[i];
  }

  // Resize the dual_code vector
  dual_code.resize(0);
  dual_code.resize(m + 1, 0);

  // Initialize a binomial cache
  std::vector<std::vector<gmpwrap::GmpRational>> binom_cache;
  init_binom_cache(binom_cache, m, num_worker_threads);

  // Atomic counter for w
  std::atomic<size_t> next_unclaimed_w{0};

  // Vector of threads
  std::vector<std::thread> worker_threads;

  // Worker threads
  for (size_t t = 0; t < num_worker_threads; ++t) {
    worker_threads.push_back(std::thread(
        [&next_unclaimed_w, &code, &dual_code, &code_size, m, &binom_cache]() {
          for (size_t w; (w = next_unclaimed_w++) <=
                         m;) {  // Claim next w directly in loop condition
            std::cout << "w = " << w << std::endl;

            // Compute dual_code[w]
            for (size_t t = 0; t <= m; ++t) {
              dual_code[w] += krawtchouk(w, t, m, binom_cache) * code[t];
            }
            dual_code[w] /= code_size;
          }
        }));
  }

  // Join all threads
  for (auto& thread : worker_threads) {
    thread.join();
  }
}

void print_weight_dist(const std::vector<gmpwrap::GmpRational>& dist) {
  for (size_t t = 0; t < dist.size(); ++t) {
    // std::cout << dist[t].to_double() << std::endl;
    std::cout << dist[t].str() << std::endl;
  }
}

#endif  // __MW_TF_HPP__
