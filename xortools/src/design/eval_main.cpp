#include <cassert>
#include <cstdint>
#include <random>

#include "argparse/argparse.hpp"
#include "design/design.hpp"

template <typename T>
void print_vec(std::ostream& out, const std::vector<T>& vec) {
  out << "{";
  for (size_t i = 0; i < vec.size(); ++i) {
    out << vec[i];
    if (i + 1 < vec.size()) {
      out << ", ";
    }
  }
  out << "}";
}

// This program evaluates a degree distribution for irregular LDPC codes under
// various quantum and classical algorithms.
int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("eval");
  program.add_argument("--verbose")
      .help("increase output verbosity")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--seed")
      .help(
          "Random seed (for reproducibility). If blank, uses a hardware-random "
          "seed.")
      .default_value(uint64_t(0))
      .scan<'u', uint64_t>();

  // Ensemble options
  program.add_argument("--degfile")
      .help(
          "Edge-perspective node and check degree distributions file (.deg "
          "format).")
      .default_value(std::string(""));
  program.add_argument("--k")
      .help("number of edges per node")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--D")
      .help("number of edges per check")
      .default_value(size_t(0))
      .scan<'u', size_t>();

  // Global experiment options
  program.add_argument("--clauses")
      .help("Number of clauses to use for the trial instances")
      .default_value(int(10000))
      .scan<'i', int>();
  program.add_argument("--samples")
      .help("Number of instances to sample for the evaluation")
      .default_value(int(100))
      .scan<'i', int>();

  // Annealing optimization experiment options
  program.add_argument("--annealing-iterations")
      .help("Number of sweeps to do annealing")
      .default_value(size_t(1'000))
      .scan<'u', size_t>();

  // Belief propagation parameters
  program.add_argument("--bp-iterations")
      .help(
          "Maximum number of iterations for belief propagation. If -1, use an "
          "automatic choice.")
      .default_value(int(-1))
      .scan<'i', int>();
  program.add_argument("--max-error-rate")
      .help("When finding the threshold, this is the maximum allowed")
      .default_value(double(0.01))
      .scan<'g', double>();
  program.add_argument("--shots")
      .help("Number of decoding trials to run per point.")
      .default_value(size_t(200))
      .scan<'u', size_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  bool verbose = program.get<bool>("verbose");
  uint64_t seed = program.get<uint64_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::cout << "using seed " << seed << std::endl;
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<uint64_t> uniform;
  auto gen = [&]() -> uint64_t { return uniform(eng); };

  // Instance ensemble options
  std::string deg_fname = program.get<std::string>("--degfile");
  size_t k = program.get<size_t>("--k");
  size_t D = program.get<size_t>("--D");
  if ((k or D) == (deg_fname.size() > 0)) {
    throw std::invalid_argument(
        "must either supply --k INT --D INT for regular codes, or use "
        "--degfile FILE.deg to specify the degree distribution.");
  }

  // Experiment options
  int num_clauses = program.get<int>("--clauses");
  int num_samples = program.get<int>("--samples");

  // Annealing optimization experiment options
  size_t num_annealing_iterations =
      program.get<size_t>("--annealing-iterations");

  // Belief propagation parameters
  int max_bp_iterations = program.get<int>("--bp-iterations");
  size_t shots_per_point = program.get<size_t>("--shots");
  double max_error_rate = program.get<double>("--max-error-rate");

  DegreeDistributionPair dist;
  double design_rate;
  if (k or D) {
    dist.node.degrees.push_back(k);
    dist.node.probabilities.push_back(1.0);
    dist.check.degrees.push_back(D);
    dist.check.probabilities.push_back(1.0);
    design_rate = 1 - double(k) / double(D);
  } else {
    assert(deg_fname.size());
    dist = load_degfile(deg_fname);
    design_rate = dist.rate();
  }

  std::cout << "𝔼 node degree = " << dist.node.vertex_perspective().mean()
            << std::endl;
  std::cout << "𝔼 check degree = " << dist.check.vertex_perspective().mean()
            << std::endl;
  std::cout << "Design rate: " << design_rate << std::endl;
  std::cout << "𝔼 Truncation value (theoretical): " << (1 - design_rate / 2)
            << std::endl;
  std::vector<double> annealing_values;
  std::cout << "𝔼 SA value (empirical) = "
            << est_saval_empirical(dist, num_clauses, num_samples,
                                   num_annealing_iterations, gen(),
                                   annealing_values)
            << std::endl;
  std::cout << "SA trial values (empirical) = ";
  print_vec(std::cout, annealing_values);
  std::cout << std::endl;

  exit(0);
  bool extras = false;
  if (extras) {
    std::vector<double> annealing_values;
    std::cout << "𝔼 SA value (empirical) = "
              << est_saval_empirical(dist, num_clauses, num_samples,
                                     num_annealing_iterations, gen(),
                                     annealing_values)
              << std::endl;
    std::cout << "𝔼 StrippedSA value (empirical) = "
              << est_stripped_saval_empirical(dist, num_clauses, num_samples,
                                              num_annealing_iterations, gen())
              << std::endl;
    double decoding_threshold_empirical = est_decoding_threshold_empirical(
        dist, num_clauses, num_samples, max_error_rate, max_bp_iterations,
        shots_per_point, gen());
    std::cout << "𝔼 Decoding threshold (empirical) = "
              << decoding_threshold_empirical << std::endl;
    std::cout << "𝔼 DQI value (based on empirical decoding threshold + "
                 "semicircle law) = "
              << est_dqival(decoding_threshold_empirical) << std::endl;
  }
  double ratio = est_quantum_advantage_over_sa_empirical(
      dist, num_clauses, num_samples, num_annealing_iterations, max_error_rate,
      max_bp_iterations, shots_per_point, gen());
  std::cout << "𝔼 quantum advantage ratio over SA and Truncation "
               "(based on empirical decoding "
               "and optimization) = "
            << ratio << std::endl;
  // double est_quantum_advantage_over_sa_empirical(DegreeDistributionPair dist,
  //                             size_t num_clauses, size_t num_samples,
  //                                     size_t num_iterations,
  //                                     double max_error_rate, size_t max_iter,
  //                                         size_t shots_per_point,
  //                                                uint64_t seed)
}
