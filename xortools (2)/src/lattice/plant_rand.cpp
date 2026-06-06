#include "argparse/argparse.hpp"
#include "bleichenbacher_nguyen.hpp"
#include "gmpwrap.hpp"
#include "sparse/sparse_matrix.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("plant_rand");
  program.add_argument("instance")
      .help("File to write the instance in .csv format to");
  program.add_argument("--p").help("Modulus").scan<'i', int>();
  program.add_argument("--m").help("Number of constraints").scan<'i', int>();
  program.add_argument("--n").help("Number of variables").scan<'i', int>();
  program.add_argument("--num-distractors")
      .help("Number of distractor points.")
      .scan<'i', int>();
  program.add_argument("--seed")
      .help(
          "Random seed (for reproducibility). If blank or 0, uses a "
          "hardware-random "
          "seed.")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--dont-add-zero")
      .help(
          "If present, do not add zero to the list (but if the "
          "randomly-sampled polynomial "
          "has a root, this will not resample the polynomial)")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--sort-targets")
      .help(
          "If present, sort the list of targets in increasing order. This "
          "effectively hides the planted solution.")
      .default_value(false)
      .implicit_value(true);
  // program.add_argument("--verbose")
  //     .help("increase output verbosity")
  //     .default_value(false)
  //     .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  std::string fname = program.get<std::string>("instance");
  int p = program.get<int>("--p");
  int m = program.get<int>("--m");
  int n = program.get<int>("--n");
  int num_distractors = program.get<int>("--num-distractors");
  // bool verbose = program.get<bool>("--verbose");
  bool dont_add_zero = program.get<bool>("--dont-add-zero");
  bool sort_targets = program.get<bool>("--sort-targets");
  uint64_t seed = program.get<uint64_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937_64 rng;
  rng.seed(seed);

  sparse::SparseMatrix B(0, 0);
  std::vector<std::vector<int>> targets;
  gen_planted_random_instance(B, targets, n, m, num_distractors, p,
                              dont_add_zero, sort_targets, rng,
                              /*rs_code=*/false);
  sparse::write_csv(B, targets, fname);
}
