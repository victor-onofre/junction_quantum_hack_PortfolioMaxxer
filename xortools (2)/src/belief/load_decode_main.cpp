#include <atomic>
#include <cassert>
#include <thread>

#include "argparse/argparse.hpp"
#include "bitmatrix.hpp"
#include "decoding.hpp"
#include "ldpc.hpp"
#include "sparse/sparse_matrix.hpp"
#include "utils.hpp"
#include "xorsat.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("load_decode_sequential");
  program.add_argument("instance").help("Instance in .tsv format");
  program.add_argument("--print-traps")
      .help("show the traps (errors that lead to a decoding failure)")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--pin-bits")
      .help("Use bit-pinning to lower the error rate")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--min-ell")
      .help(
          "Minimum number of bit-flip errors to consider for the search range")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--max-ell")
      .help(
          "Maximum number of bit-flip errors to consider for the search range")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--min-e")
      .help("Minimum number of erasure errors to consider for the search range")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--max-e")
      .help("Maximum number of erasure errors to consider for the search range")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--shots")
      .help("Number of decoding trials to run per point.")
      .default_value(size_t(20))
      .scan<'u', size_t>();
  program.add_argument("--binary-search")
      .help(
          "If only one of --max-ell, --max-e is nonzero, can do a binary "
          "search to find the threshold more quickly.")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--seed")
      .help(
          "Random seed (for reproducibility). If blank, uses a hardware-random "
          "seed.")
      .default_value(uint64_t(0))
      .scan<'u', uint64_t>();
  program.add_argument("--maxiter")
      .help(
          "Maximum number of iterations for belief propagation. If -1, use an "
          "automatic choice.")
      .default_value(int(-1))
      .scan<'i', int>();
  program.add_argument("--threads")
      .help(
          "Number of threads to use for decoding (0 means use the hardware "
          "concurrency)")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--max-error-rate")
      .help("When finding the threshold, this is the maximum allowed")
      .default_value(double(0.01))
      .scan<'g', double>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  size_t shots_per_point = program.get<size_t>("--shots");

  uint64_t seed = program.get<uint64_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::cout << "using seed " << seed << "\n";
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<unsigned> uniform;
  auto gen = [&]() { return uniform(eng); };

  std::string fname = program.get<std::string>("instance");
  bool do_binary_search = program.get<bool>("--binary-search");

  bitmatrix H;
  {
    instance I;
    if (!load_instance(fname, I)) {
      throw std::invalid_argument("Malformed / missing tsv file: " + fname);
    }
    H = parity_check_matrix(I);
  }

  bool print_traps = program.get<bool>("--print-traps");
  bool pin_bits = program.get<bool>("--pin-bits");
  size_t min_ell = program.get<size_t>("--min-ell");
  size_t max_ell = program.get<size_t>("--max-ell");
  size_t min_e = program.get<size_t>("--min-e");
  size_t max_e = program.get<size_t>("--max-e");
  int maxiter = program.get<int>("--maxiter");
  size_t num_threads = program.get<size_t>("--threads");
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }
  double max_decoding_error_rate = program.get<double>("--max-error-rate");
  struct ErrorRate {
    double error_rate;
    double bit_error_rate;
    size_t num_shots;
  };
  auto get_error_rate = [&](size_t num_shots, size_t ell,
                            size_t e) -> ErrorRate {
    // const size_t num_threads = 1;
    std::atomic<size_t> num_shots_claimed = 0;
    std::atomic<size_t> num_shots_completed = 0;
    std::atomic<size_t> num_errors = 0;
    std::atomic<size_t> num_error_bits = 0;
    std::vector<std::thread> threads;
    std::vector<std::atomic<size_t>> trap_counts(H.num_cols());
    double unerased_p_flip = double(ell) / double(H.num_cols());
    // double unerased_p_flip = 10.0 / double(H.num_cols());
    for (size_t t = 0; t < num_threads; ++t) {
      threads.push_back(std::thread(
          [&print_traps, &pin_bits, &trap_counts, &num_shots_claimed,
           &num_shots_completed, &num_errors, &num_error_bits, &maxiter,
           &unerased_p_flip, &H, &ell, &e, &num_shots](uint64_t thread_seed) {
            std::mt19937_64 thread_eng(thread_seed);
            std::uniform_real_distribution<double> uniform_unit{
                0.0, std::nextafter(1.0, 2.0)};
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
              // bitvector corrupted_codeword =
              //     apply_rand_err(random_codeword, thread_eng, ell);
              std::vector<size_t> bitflip_errors = random_subset<size_t>(
                  random_codeword.size(), ell, thread_eng);
              bitvector corrupted_codeword = random_codeword;
              for (const auto& i : bitflip_errors) corrupted_codeword.flip(i);

              // Apply erasure errors
              if (e > 0) {
                bitvector erased = apply_rand_err(erasures, thread_eng, e);
                for (size_t i = 0; i < H.num_cols(); ++i) {
                  if (erased.get(i)) {
                    p_flip[i] = 0.5;
                    corrupted_codeword.set(i, 0);
                  }
                }
              }

              bitvector decoded;
              if (pin_bits) {
                bool found_sol = false;
                // std::vector<size_t> bits_to_pin = {20, 94, 6, 45, 60};
                // std::vector<size_t> bits_to_pin =
                // {33,49,73,389,109,260,372,61};
                // std::vector<size_t> bits_to_pin = {917, 911, 876, 887, 800,
                // 576, 892, 867, 897};
                // std::vector<size_t> bits_to_pin = {2999};
                // std::vector<size_t> bits_to_pin = {2881, 791, 815, 2347,
                // 1174, 863};
                std::vector<size_t> bits_to_pin = {648};
                // std::vector<size_t> bits_to_pin = {579,4199};
                // std::vector<size_t> bits_to_pin = {7436};
                for (size_t r = 0; !found_sol and r <= bits_to_pin.size();
                     ++r) {
                  // Iterate over combinations of r pinnable bits
                  std::vector<bool> pinned(bits_to_pin.size(), false);
                  std::fill(pinned.begin() + pinned.size() - r, pinned.end(),
                            true);
                  do {
                    // Set the pinned bit p_flips to 1 and the other bits to 0
                    for (size_t i = 0; i < pinned.size(); ++i) {
                      p_flip[bits_to_pin[i]] = pinned[i];
                    }
                    // Try decoding with the current pinning
                    decoded =
                        dec.bp_decode(corrupted_codeword, p_flip, maxiter);
                    bool is_codeword = ((decoded * dec.HT).count() == 0);
                    // if (decoded == random_codeword and is_codeword) {
                    //   found_sol = true;
                    // }
                    if (is_codeword and
                        (decoded + corrupted_codeword).count() <= ell) {
                      // As soon as we find an error pattern of size at most
                      // ell, we terminate
                      if (decoded != random_codeword) {
                        bitvector delta = (decoded + random_codeword);
                        std::cout << "found a weight " << delta.count()
                                  << " codeword: " << delta << std::endl;
                        assert((delta * dec.HT).count() == 0);
                      }
                      found_sol = true;
                    }
                  } while (!found_sol and
                           std::next_permutation(pinned.begin(), pinned.end()));
                }
              } else {
                // Decode one time with no pinning
                decoded = dec.bp_decode(corrupted_codeword, p_flip, maxiter);
              }

              bool success = (decoded == random_codeword);
              num_error_bits += (decoded + random_codeword).count();
              if (!success) {
                ++num_errors;
                if (print_traps) {
                  std::cout << "trapping set: { ";
                  for (size_t i = 0; i < bitflip_errors.size(); ++i) {
                    if (i > 0) std::cout << " ,  ";
                    std::cout << bitflip_errors[i];
                    ++trap_counts[bitflip_errors[i]];
                  }
                  std::cout << " }" << std::endl;
                  std::vector<size_t> idx = argsort(trap_counts);
                  std::cout << "current trapped counts ";
                  size_t count = 0;
                  for (auto it = idx.rbegin(); it != idx.rend(); ++it) {
                    if (trap_counts[*it] <= 1 or count++ > 100) break;
                    std::cout << " " << *it << " appears " << trap_counts[*it]
                              << " times;";
                  }
                  std::cout << std::endl;
                  std::cout << "num_errors = " << num_errors
                            << " num_shots_claimed = " << num_shots_claimed
                            << " num_shots_completed = " << num_shots_completed
                            << std::endl;
                }
              }

              if (e > 0) {
                p_flip.resize(0);
                p_flip.resize(H.num_cols(), unerased_p_flip);
              }
            }
          },
          /*thread_seed=*/gen()));
    }
    for (size_t t = 0; t < num_threads; ++t) {
      threads[t].join();
    }
    ErrorRate result;
    result.error_rate = double(num_errors) / double(num_shots_claimed);
    result.bit_error_rate =
        double(num_error_bits) / double(H.num_cols() * num_shots_claimed);
    result.num_shots = num_shots_claimed;
    return result;
  };

  if (do_binary_search) {
    size_t ell = 0;
    size_t e = 0;
    if ((max_ell == 0) and (max_e == 0)) {
      // By default, assume user wants to find the bitflip decoding threshold
      max_ell = maximum_possible_ell(H.num_rows(), H.num_cols());
    }
    if ((max_ell != 0) == (max_e != 0))
      throw std::invalid_argument(
          "In binary search mode, can only provide one of --max-ell, --max-e.");

    if ((min_ell != 0) and (min_e != 0))
      throw std::invalid_argument(
          "In binary search mode, can only provide one of --min-ell, --min-e.");

    bool do_erasures = (max_e > 0);
    size_t lower = min_ell | min_e;
    size_t upper = max_ell | max_e;
    if (upper == 0) upper = H.num_cols();
    while (lower + 1 < upper) {
      int error_param = (lower + upper) >> 1;
      if (do_erasures) {
        e = error_param;
      } else {
        ell = error_param;
      }
      ErrorRate result = get_error_rate(shots_per_point, ell, e);

      std::cout << "ell = " << ell << " lo = " << lower << " hi = " << upper
                << " e = " << e << " m = " << H.num_cols()
                << " error_rate = " << result.error_rate
                << " bit_error_rate = " << result.bit_error_rate
                << " num_shots = " << result.num_shots << std::endl;
      if (result.error_rate > max_decoding_error_rate) {
        upper = error_param;
      } else {
        lower = error_param;
      }
    }
    std::cout << "max error parameter = " << lower << "\n";
  } else {
    // Linear search mode
    for (size_t e = min_e; e <= max_e; ++e) {
      for (size_t ell = min_ell; ell <= max_ell; ++ell) {
        for (size_t v = 0; v < 1; ++v) {
          ErrorRate result = get_error_rate(shots_per_point, ell, e);
          std::cout << "ell = " << ell << " e = " << e
                    << " m = " << H.num_cols()
                    << " error_rate = " << result.error_rate
                    << " bit_error_rate = " << result.bit_error_rate
                    << " num_shots = " << result.num_shots << std::endl;
        }
      }
    }
  }
}
