#include <cassert>
#include <random>

#include "argparse/argparse.hpp"
#include "optimize/anneal.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("irreg_anneal");
  program.add_argument("instance").help("Instance in .tsv format");
  program.add_argument("--seed")
      .help(
          "Random seed (for reproducibility). If blank or 0, uses a "
          "hardware-random "
          "seed.")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--iterations")
      .help("Number of sweeps to do annealing")
      .default_value(size_t(1'000'000))
      .scan<'u', size_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  uint64_t seed = program.get<uint64_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::cout << "using seed " << seed << "\n";
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<uint64_t> uniform;
  auto gen = [&]() { return uniform(eng); };

  size_t iterations = program.get<size_t>("--iterations");
  std::string fname = program.get<std::string>("instance");
  std::vector<int> target;
  sparse::SparseMatrix H(0, 0);
  sparse::read_tsv(H, target, fname);

  bool verbose = true;
  std::vector<bool> x = irreg_anneal(H, target, iterations, gen, verbose);
  int num_sat = evaluate(H, target, x);
  std::cout << "final x = ";
  for (size_t i = 0; i < x.size(); ++i) {
    std::cout << int(x[i]);
  }
  std::cout << "\nfinal number of satisfied clauses = " << num_sat
            << " (out of " << H.ncols << ")\n";
}
