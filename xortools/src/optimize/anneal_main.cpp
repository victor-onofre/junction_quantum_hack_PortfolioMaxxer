#include <cassert>
#include <random>

#include "argparse/argparse.hpp"
#include "optimize/anneal.hpp"

void parse_bitstring(const std::string& s, std::vector<bool>& x) {
  for (char c : s) {
    if (c == '0') {
      x.push_back(1);
    } else {
      assert(c == '1');
      x.push_back(0);
    };
  }
  std::cout << "x.size() = " << x.size() << "\n";
}

// This runs plain vanilla simulated annealing on the instance.
int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("anneal");
  program.add_argument("--verbose")
      .help("increase output verbosity")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("instance").help("Instance in .tsv format");
  program.add_argument("--x")
      .help("initial solution bitstring to use for optimization")
      .default_value(std::string(""));
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
  program.add_argument("--min-beta")
      .help("Initial beta to use during the sweeps.")
      .default_value(double(0))
      .scan<'g', double>();
  program.add_argument("--max-beta")
      .help("Max beta to reach.")
      .default_value(double(5))
      .scan<'g', double>();
  program.add_argument("--min-acceptance-rate")
      .help(
          "Annealing ends early if the acceptance rate drops below this "
          "number. (default 0)")
      .default_value(double(0))
      .scan<'g', double>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  bool verbose = program.get<bool>("verbose");
  size_t seed = program.get<size_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::cout << "using seed " << seed << "\n";
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<uint64_t> uniform;
  auto gen = [&]() { return uniform(eng); };

  const size_t iterations = program.get<size_t>("--iterations");
  const double min_beta = program.get<double>("--min-beta");
  const double max_beta = program.get<double>("--max-beta");
  std::cout << "min_beta = " << min_beta << " max_beta = " << max_beta
            << " iterations = " << iterations << "\n";
  double min_acceptance_rate = program.get<double>("--min-acceptance-rate");
  std::string fname = program.get<std::string>("instance");
  std::vector<int> target;
  sparse::SparseMatrix H(0, 0);
  sparse::read_tsv(H, target, fname);
  std::string initial_x_string = program.get<std::string>("--x");
  std::vector<bool> x;
  if (initial_x_string.size() == 0) {
    // This samples a random starting point
    x = anneal(H, target, iterations, min_beta, max_beta, min_acceptance_rate,
               gen, verbose);
  } else {
    // Use the provided x as a starting point
    std::vector<bool> initial_x;
    parse_bitstring(initial_x_string, initial_x);
    x = anneal(H, target, initial_x, iterations, min_beta, max_beta,
               min_acceptance_rate, gen, verbose);
  }
  int num_sat = evaluate(H, target, x);
  std::cout << "final x = ";
  for (size_t i = 0; i < x.size(); ++i) {
    std::cout << int(x[i]);
  }
  std::cout << "\nfinal number of satisfied clauses = " << num_sat
            << " (out of " << H.ncols << ")\n";
}
