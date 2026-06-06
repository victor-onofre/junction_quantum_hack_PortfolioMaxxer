#include "decoding.hpp"

#include <atomic>
#include <cassert>
#include <thread>

#include "ldpc.hpp"
#include "xorsat.hpp"

bitmatrix parity_check_bitmatrix(const sparse::SparseMatrix& H) {
  bitmatrix B(H.nrows, H.ncols);
  for (const sparse::SparseMatrix::Entry& entry : H.entries) {
    assert(entry.val == 1);
    B.set(entry.row, entry.col, 1);
  }
  return B;
}

// Function to compute the entropy of a probability
double binary_entropy(double p) {
  if (p == 0 || p == 1) {
    return 0;  // Avoid log(0)
  } else {
    return -p * std::log2(p) - (1 - p) * std::log2(1 - p);
  }
}

// Function to compute the inverse binary entropy using bisection method
double inverse_binary_entropy(double rate, double tolerance) {
  double low = 0, high = 0.5;  // Binary entropy is symmetric around 0.5
  double mid = (low + high) / 2;
  double entropy_mid;

  // TODO fix the bug
  while (high - low > tolerance) {
    mid = (low + high) / 2;
    entropy_mid = binary_entropy(mid);

    if (entropy_mid > rate) {
      high = mid;  // Entropy is too high, decrease p
    } else {
      low = mid;  // Entropy is low enough, increase p
    }
  }

  return mid;
}

size_t maximum_possible_ell(double num_rows, double num_cols) {
  return size_t(num_cols * inverse_binary_entropy(num_rows / num_cols) + 1);
}

// maxiter == -1 means choose maxiter automatically
double get_max_tolerable_error_rate(const sparse::SparseMatrix& B,
                                    double max_decoding_error_rate,
                                    size_t shots_per_point, int maxiter,
                                    size_t num_threads, std::mt19937_64& eng,
                                    bool verbose) {
  std::uniform_int_distribution<unsigned> uniform;
  auto gen = [&]() { return uniform(eng); };
  bitmatrix H = parity_check_bitmatrix(B);
  bp dec(H);

  size_t max_ell = maximum_possible_ell(H.num_rows(), H.num_cols());

  auto get_error_rate = [&](size_t num_shots,
                            size_t ell) -> std::pair<double, size_t> {
    // const size_t num_threads = 1;
    std::atomic<size_t> num_shots_claimed = 0;
    std::atomic<size_t> num_errors = 0;
    std::vector<std::thread> threads;
    double unerased_p_flip = double(ell) / double(H.num_cols());
    for (size_t t = 0; t < num_threads; ++t) {
      threads.push_back(std::thread(
          [&num_shots_claimed, &num_errors, &maxiter, &unerased_p_flip, &H,
           &ell, &num_shots](uint64_t thread_seed) {
            std::mt19937_64 thread_eng(thread_seed);
            bp dec(H);
            bitvector random_codeword;
            // not so random, but definitely a codeword
            random_codeword.zeros(H.num_cols());
            bitvector erasures;
            erasures.zeros(H.num_cols());
            std::vector<double> p_flip(H.num_cols(), unerased_p_flip);
            while (num_shots_claimed < num_shots) {
              ++num_shots_claimed;

              // Apply bit-flip errors
              bitvector corrupted_codeword =
                  apply_rand_err(random_codeword, thread_eng, ell);

              // Decode
              bitvector decoded =
                  dec.bp_decode(corrupted_codeword, p_flip, maxiter);
              if (decoded != random_codeword) ++num_errors;
            }
          },
          /*thread_seed=*/gen()));
    }
    for (size_t t = 0; t < num_threads; ++t) {
      threads[t].join();
    }

    return {double(num_errors) / double(num_shots_claimed), num_shots_claimed};
  };

  size_t ell = 0;

  size_t lower = 0;
  size_t upper = max_ell;
  if (upper == 0) upper = H.num_cols();
  while (lower + 1 < upper) {
    size_t error_param = (lower + upper) >> 1;
    ell = error_param;
    auto [error_rate, num_shots_finished] =
        get_error_rate(shots_per_point, ell);
    if (verbose) {
      std::cout << "lo = " << lower << " hi = " << upper << " ell = " << ell
                << " m = " << H.num_cols() << " error_rate = " << error_rate
                << " num_shots = " << num_shots_finished << "\n";
    }
    if (error_rate > max_decoding_error_rate) {
      upper = error_param;
    } else {
      lower = error_param;
    }
  }
  if (verbose) {
    std::cout << "max error parameter = " << lower << "\n";
  }
  return double(lower) / double(H.num_cols());
}
