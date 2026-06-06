#include <map>
#include <string>

#include "argparse/argparse.hpp"
#include "bleichenbacher_nguyen.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("bleichenbacher_nguyen");
  program.add_argument("--instance")
      .help("Instance in .csv format")
      .default_value(std::string(""));
  program.add_argument("--verbose")
      .help("increase output verbosity")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--sweep-num-distractors")
      .help(
          "If present, tries all possible number of distractors. Otherwise "
          "tries only (p-3)/2")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--sweep-m")
      .help(
          "If present, tries all possible number of clauses (m). Otherwise "
          "tries only m = p-1")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--lll")
      .help("use LLL basis reduction")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--bkz")
      .help("use BKZ basis reduction")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--same-sum")
      .help("apply integer same-sum constraint to solution lattice")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--max-block-size")
      .help(
          "Default 0 which means no limit on the block size. Limits the block "
          "size used in BKZ so that the algorithm takes polynomial time. Note "
          "that the runtime is exponential in the block size.")
      .default_value(int(0))
      .scan<'i', int>();
  program.add_argument("--seed")
      .help(
          "Random seed (for reproducibility). If blank or 0, uses a "
          "hardware-random "
          "seed.")
      .default_value(size_t(0))
      .scan<'u', size_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  bool verbose = program.get<bool>("--verbose");
  bool sweep_num_distractors = program.get<bool>("--sweep-num-distractors");
  bool sweep_m = program.get<bool>("--sweep-m");
  bool use_lll = program.get<bool>("--lll");
  bool use_bkz = program.get<bool>("--bkz");
  bool apply_same_sum_constraint = program.get<bool>("--same-sum");
  if (!use_lll and !use_bkz) {
    throw std::invalid_argument("Must provide at least one of --lll, --bkz");
  }
  std::string fname = program.get<std::string>("--instance");
  int max_block_size = program.get<int>("--max-block-size");
  uint64_t seed = program.get<uint64_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937_64 rng;
  rng.seed(seed);
  if (fname.size() > 0) {
    // Load the problem instance from a file
    std::vector<std::vector<int>> targets;
    sparse::SparseMatrix B(0, 0);
    sparse::read_csv(B, targets, fname);

    auto result =
        bleichenbacher_nguyen(B, targets, verbose, use_lll, use_bkz,
                              max_block_size, apply_same_sum_constraint);
    std::cout << "got solution with num_sat = " << result.best_num_sat
              << " smallest L2 norm √" << result.min_l2_squared
              << " smallest L1 norm " << result.min_l1 << " for " << fname
              << "\n";
    return 0;
  }

  // Generate problem instances randomly
  int num_shots_total = 2;
  std::cout << "p,apply_same_sum,m,n,num_distractors,max_num_sat,min_l1,min_"
               "l22,num_shots\n";
  for (int p : {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}) {
    if (sweep_num_distractors and p > 29) continue;
    if (sweep_num_distractors and p > 17 and use_bkz and max_block_size == 0)
      continue;
    if (use_bkz and ((max_block_size == 0 and p > 29) or p > 37)) continue;
    std::vector<int> num_distractors_to_try;
    if (sweep_num_distractors) {
      for (int num_distractors = 0; num_distractors <= (p - 1);
           ++num_distractors) {
        num_distractors_to_try.push_back(num_distractors);
      }
    } else {
      // Just try (p-3)/2 corresponding to the OPI regime
      num_distractors_to_try.push_back((p - 3) / 2);
    }
    std::vector<int> m_to_try;
    if (sweep_m) {
      for (int m = 2; m < p; ++m) {
        m_to_try.push_back(m);
      }
    } else {
      m_to_try.push_back(p - 1);
    }
    for (bool apply_same_sum_constraint : {true, false}) {
      for (int num_distractors : num_distractors_to_try) {
        for (int m : m_to_try) {
          int n = m >> 1;
          std::map<bool, std::map<int, std::map<int, std::map<int, int>>>>
              apply_same_sum_best_num_sat_min_L1_min_L22_to_num_shots;
          for (int shot = 0; shot < num_shots_total; ++shot) {
            std::vector<std::vector<int>> targets;
            sparse::SparseMatrix B(0, 0);
            gen_planted_random_instance(B, targets, n, m, num_distractors, p,
                                        /*dont_add_zero=*/false,
                                        /*sort_targets=*/true, rng,
                                        /*rs_code=*/true);
            auto result = bleichenbacher_nguyen(
                B, targets, /*verbose=*/false, use_lll, use_bkz, max_block_size,
                apply_same_sum_constraint);
            ++apply_same_sum_best_num_sat_min_L1_min_L22_to_num_shots
                [apply_same_sum_constraint][result.best_num_sat][result.min_l1]
                [result.min_l2_squared];
          }
          for (auto& [apply_same_sum, it0] :
               apply_same_sum_best_num_sat_min_L1_min_L22_to_num_shots) {
            for (auto& [best_num_sat, it1] : it0) {
              for (auto& [min_l1, it2] : it1) {
                for (auto& [min_l22, num_shots] : it2) {
                  std::cout << p << "," << apply_same_sum << "," << m << ","
                            << n << "," << num_distractors << ","
                            << best_num_sat << "," << min_l1 << "," << min_l22
                            << "," << num_shots << "\n";
                }
              }
            }
          }
        }
      }
    }
  }
}
