#include "macwilliams_transform.hpp"

#include <gtest/gtest.h>

#include <memory>

// Function to compute Krawtchouk polynomials using the recurrence relation with
// parallelization
std::vector<std::vector<gmpwrap::GmpRational>> compute_krawtchouk_recurrence(
    size_t m, size_t max_num_threads) {
  std::vector<std::vector<gmpwrap::GmpRational>> table(
      m + 1, std::vector<gmpwrap::GmpRational>(m + 1, gmpwrap::GmpRational(0)));

  constexpr bool verbose = false;

  // Replace this section for initializing completion_status
  std::vector<std::vector<std::unique_ptr<std::atomic<bool>>>>
      completion_status(m + 1);

  // Initialize completion status
  for (size_t w = 0; w <= m; ++w) {
    completion_status[w].resize(m + 1);
    for (size_t t = 0; t <= m; ++t) {
      completion_status[w][t] = std::make_unique<std::atomic<bool>>(false);
    }
  }

  // Fill the top row (w=0) with ones
  for (size_t t = 0; t <= m; ++t) {
    table[0][t] = gmpwrap::GmpRational(1);
    if (verbose) {
      std::cout << "completing w = " << 0 << ", t = " << t << " offline\n";
    }
    completion_status[0][t]->store(true, std::memory_order_release);
  }

  // Fill the rightmost column (t=m) with alternating binomial coefficients
  for (size_t w = 0; w <= m; ++w) {
    table[w][m] =
        ((w % 2 == 0) ? gmpwrap::GmpRational(1) : gmpwrap::GmpRational(-1)) *
        binom(m, w);
    if (verbose) {
      std::cout << "completing w = " << w << ", t = " << m << " offline\n";
    }
    completion_status[w][m]->store(true, std::memory_order_release);
  }

  // Create a topologically-ordered list of tasks for the main table entries
  // (zigzagging for maximum parallelism)
  std::vector<std::pair<size_t, size_t>> tasks;
  for (size_t t = m - 1; t < m; --t) {  // Iterate columns from right to left
    for (size_t w = 1; w <= m; ++w) {   // Iterate rows from top to bottom
      tasks.emplace_back(w, t);
    }
  }

  // Atomic counter for task claims
  std::atomic<size_t> next_task{0};

  // Worker threads
  std::vector<std::thread> worker_threads;
  for (size_t thread_id = 0; thread_id < max_num_threads; ++thread_id) {
    worker_threads.emplace_back([&]() {
      size_t task_index;
      while ((task_index = next_task++) < tasks.size()) {
        const auto& [w, t] = tasks[task_index];
        if (verbose) {
          std::cout << "completing task: w = " << w << " t = " << t
                    << std::endl;
          std::cout << "dependencies: " << w - 1 << ", " << t + 1 << "; "
                    << w - 1 << ", " << t << "; " << w << ", " << t + 1
                    << std::endl;
        }
        // Spinlock waiting for dependencies
        while (!completion_status.at(w - 1).at(t + 1)->load(
                   std::memory_order_acquire) ||
               !completion_status.at(w - 1).at(t)->load(
                   std::memory_order_acquire) ||
               !completion_status[w][t + 1]->load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }

        // Compute the current table entry
        table[w][t] = table[w - 1][t + 1] + table[w - 1][t] + table[w][t + 1];
        completion_status[w][t]->store(true, std::memory_order_release);
      }
    });
  }

  // Join all threads
  for (auto& thread : worker_threads) {
    thread.join();
  }

  return table;
}

void test_krawtchouk_table(size_t m, size_t max_num_threads) {
  // Compute Krawtchouk polynomials using the provided function
  std::vector<std::vector<gmpwrap::GmpRational>> krawtchouk_direct(
      m + 1, std::vector<gmpwrap::GmpRational>(m + 1));
  std::vector<std::vector<gmpwrap::GmpRational>> binom_cache;
  init_binom_cache(binom_cache, m, std::thread::hardware_concurrency());

  for (size_t w = 0; w <= m; ++w) {
    for (size_t t = 0; t <= m; ++t) {
      krawtchouk_direct[w][t] = krawtchouk(w, t, m, binom_cache);
    }
  }

  // Compute Krawtchouk polynomials using the recurrence relation
  auto krawtchouk_recurrence =
      compute_krawtchouk_recurrence(m, max_num_threads);

  // Compare results
  for (size_t w = 0; w <= m; ++w) {
    for (size_t t = 0; t <= m; ++t) {
      EXPECT_EQ(krawtchouk_direct[w][t].str(),
                krawtchouk_recurrence[w][t].str())
          << "Mismatch at (w, t) = (" << w << ", " << t << ")";
    }
  }
}

// Test for comparing the two methods of computing Krawtchouk polynomials
TEST(MacWilliamsTransformTest, CompareKrawtchoukMethods) {
  constexpr size_t max_num_threads = 10;
  test_krawtchouk_table(3, max_num_threads);
  test_krawtchouk_table(10, max_num_threads);
  test_krawtchouk_table(11, max_num_threads);
  test_krawtchouk_table(20, max_num_threads);
  test_krawtchouk_table(100, max_num_threads);
  test_krawtchouk_table(101, max_num_threads);
}
