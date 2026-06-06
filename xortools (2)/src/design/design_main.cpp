#include <cassert>
#include <fstream>
#include <random>

#include "argparse/argparse.hpp"
#include "belief/density.hpp"
#include "design.hpp"
#include "irreg/distribution.hpp"
#include "irreg/irreg.hpp"
#include "optimize/problem.hpp"
#include "sparse/sparse_matrix.hpp"

// TODO: once 1.85.0 is supported by rules_bazel, remove this hack
#include "deps/boost_optimization_1_85_0/differential_evolution.hpp"
#include "deps/boost_optimization_1_85_0/random_search.hpp"

// This program designs a degree distribution for irregular LDPC codes that is
// optimized to have a high threshold under density evolution and meet some rate
// and degree constraints.
int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("design");
  program.add_argument("--min-rate")
      .help(
          "The design rate of the irregular LDPC code degree distribution. "
          "This is used as a lower bound, so the rate of the distribution "
          "found could actually be slightly higher than this.")
      .scan<'g', double>();
  program.add_argument("--mutation")
      .help("Mutation factor to use for differentail evolution.")
      .default_value(double(0.01))
      .scan<'g', double>();
  program.add_argument("--crossover")
      .help("Crossover probability to use for differentail evolution.")
      .default_value(double(0.5))
      .scan<'g', double>();
  program.add_argument("--delta-max")
      .help(
          "The upper bound for the binary search to find the threshold value. "
          "Default is 0.5, but lowering this can improve speed.")
      .default_value(double(0.5))
      .scan<'g', double>();
  program.add_argument("--verbose")
      .help("increase output verbosity")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--optimize-advantage")
      .help(
          "instead of optimizing delta for a fixed rate, this mode switches it "
          "to just optimize for the estimated relative factor of quantum "
          "advantage")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--deg-fname")
      .help(
          "The name of the degree distribution file to put the designed "
          "distribution. This will be periodically updated as better "
          "distributions are found.");
  program.add_argument("--initial-deg-fname")
      .help(
          "The name of a degree distribution file used as the template and "
          "initial state of the degree distribution to optimize. If this is "
          "present, then --k-initial, --d-initial and "
          "--{min,max}-{node,check}-degree are ignored.")
      .default_value(std::string(""));
  program.add_argument("--max-node-degree")
      .help("maximum degree of the nodes")
      .default_value(size_t(100))
      .scan<'u', size_t>();
  program.add_argument("--max-check-degree")
      .help("maximum degree of the checks")
      .default_value(size_t(100))
      .scan<'u', size_t>();
  program.add_argument("--min-node-degree")
      .help("minimum degree of the nodes")
      .default_value(size_t(2))
      .scan<'u', size_t>();
  program.add_argument("--min-check-degree")
      .help("minimum degree of the checks")
      .default_value(size_t(2))
      .scan<'u', size_t>();
  program.add_argument("--k-initial")
      .help(
          "initially, the distribution is set to a (k,D) regular code where "
          "the k and D are given by these initial values")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--D-initial")
      .help("see --k-initial")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--bins")
      .help(
          "Number of LLR bins to use for quantization in density evolution "
          "(must be odd)")
      .default_value(size_t(151))
      .scan<'u', size_t>();
  program.add_argument("--density-iterations")
      .help("Number of BP iterations to model (default 1000)")
      .default_value(size_t(100))
      .scan<'u', size_t>();
  program.add_argument("--shrink-factor")
      .help(
          "Shrinking the bit error rate by this factor is interpreted as "
          "indicating below threshold.")
      .default_value(double(0.5))
      .scan<'g', double>();
  program.add_argument("--threads")
      .help(
          "Number of threads to spawn for the evaluation of the fitness "
          "function")
      .default_value(size_t(1))
      .scan<'u', size_t>();
  program.add_argument("--pop", "--population-size")
      .help(
          "Number of members to maintain in the population during differential "
          "evolution.")
      .default_value(size_t(15))
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

  // Belief propagation experiment options
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
      .default_value(size_t(20))
      .scan<'u', size_t>();
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
  bool opt_advantage = program.get<bool>("--optimize-advantage");

  // Code family constraints
  double min_rate = program.get<double>("--min-rate");
  double mutation_factor = program.get<double>("--mutation");
  double crossover_probability = program.get<double>("--crossover");
  std::string deg_fname = program.get<std::string>("--deg-fname");
  double delta_min = 1e-10;
  double delta_max = program.get<double>("--delta-max");
  size_t max_node_degree = program.get<size_t>("--max-node-degree");
  size_t max_check_degree = program.get<size_t>("--max-check-degree");
  size_t min_node_degree = program.get<size_t>("--min-node-degree");
  size_t min_check_degree = program.get<size_t>("--min-check-degree");

  // Density evolution parameters
  double precision = 0.01;
  size_t bins = program.get<size_t>("--bins");
  size_t num_density_iterations = program.get<size_t>("--density-iterations");
  double shrink_factor = program.get<double>("--shrink-factor");

  // Differential evolution parameters
  size_t num_threads = program.get<size_t>("--threads");
  size_t population_size = program.get<size_t>("--pop");

  // Global experiment options
  int num_clauses = program.get<int>("--clauses");
  int num_samples = program.get<int>("--samples");

  // Annealing optimization experiment options
  size_t num_annealing_iterations =
      program.get<size_t>("--annealing-iterations");

  // Belief propagation parameters
  int max_bp_iterations = program.get<int>("--bp-iterations");
  size_t shots_per_point = program.get<size_t>("--shots");
  double max_error_rate = program.get<double>("--max-error-rate");

  size_t seed = program.get<size_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937_64 eng;
  eng.seed(seed);
  std::uniform_int_distribution<uint64_t> uniform;
  auto gen = [&]() -> uint64_t { return uniform(eng); };

  // Initialize the degree distribution
  size_t k_initial = program.get<size_t>("--k-initial");
  size_t D_initial = program.get<size_t>("--D-initial");
  std::string initial_deg_fname =
      program.get<std::string>("--initial-deg-fname");
  DegreeDistributionPair dist;
  if (initial_deg_fname.size()) {
    if (k_initial or D_initial) {
      throw std::invalid_argument(
          "Must provide EITHER --initial-deg-fname OR --k-initial and "
          "--D-initial to set the initial degree distribution.");
    }
    dist = load_degfile(initial_deg_fname);
  } else {
    // Initialize using k_initial and D_initial
    if (k_initial < 2 or D_initial < 2) {
      throw std::invalid_argument("Must have k, D >= 2");
    }
    for (size_t d = min_node_degree; d <= max_node_degree; ++d) {
      dist.node.degrees.push_back(d);
      dist.node.probabilities.push_back(d == k_initial);
    }
    for (size_t d = min_check_degree; d <= max_check_degree; ++d) {
      dist.check.degrees.push_back(d);
      dist.check.probabilities.push_back(d == D_initial);
    }
  }

  auto write = [deg_fname](DegreeDistributionPair& dist) {
    std::ofstream outfile;
    outfile.open(deg_fname);
    if (!outfile.is_open()) {
      throw std::runtime_error("Unable to create output file");
    }
    outfile << dist.degfile_str();
  };

  std::function<double(const DegreeDistributionPair&)> obj;
  std::mutex random_mutex;
  if (opt_advantage) {
    obj = [num_clauses, num_samples, num_annealing_iterations, max_error_rate,
           max_bp_iterations, shots_per_point, &random_mutex,
           &gen](const DegreeDistributionPair& mutated) {
      // auto [lo, hi] = binary_search_threshold(
      //     mutated, delta_min, delta_max, precision, bins, num_iterations,
      //     /*fractional_quantization=*/false, /*early_stop=*/true,
      //     /*quantization_fiddling=*/false, /*shrink_factor=*/shrink_factor,
      //     verbose, /*halt_if_leq=*/0);
      // double mutated_delta = lo;
      // double mutated_delta = est_decoding_threshold(mutated);
      // double mutated_dqival = est_dqival(mutated_delta);
      // // double mutated_classical =
      // //     std::max(est_thval(mutated),
      // est_adaptive_saval_empirical(mutated, gen())); double mutated_classical
      // =
      //     std::max(est_thval(mutated), est_stripped_saval_empirical(mutated,
      //     gen()));
      // double mutated_advantage = mutated_dqival / mutated_classical;
      uint64_t thread_seed;
      {
        std::lock_guard<std::mutex> lock(random_mutex);
        thread_seed = gen();
      }
      double mutated_advantage = est_quantum_advantage_over_sa_empirical(
          mutated, num_clauses, num_samples, num_annealing_iterations,
          max_error_rate, max_bp_iterations, shots_per_point, thread_seed);
      return -mutated_advantage;
    };
  } else {
    obj = [delta_min, delta_max, precision, bins, num_density_iterations,
           shrink_factor, verbose](const DegreeDistributionPair& mutated) {
      auto [lo, hi] = binary_search_threshold(
          mutated, delta_min, delta_max, precision, bins,
          num_density_iterations,
          /*fractional_quantization=*/false, /*early_stop=*/true,
          /*quantization_fiddling=*/false, /*shrink_factor=*/shrink_factor,
          verbose, /*halt_if_leq=*/0);

      return -lo;
    };
  }

  auto vec_to_dist =
      [&](const std::vector<double>& x) -> DegreeDistributionPair {
    // Handles conversion from vector to degree distribution pair.
    size_t counter = 0;
    DegreeDistributionPair tmp_dist;
    assert(dist.check.degrees.size() + dist.node.degrees.size() == x.size());
    for (size_t i = 0; i < dist.check.degrees.size(); ++i) {
      tmp_dist.check.degrees.push_back(dist.check.degrees.at(i));
      tmp_dist.check.probabilities.push_back(x.at(counter));
      ++counter;
    }
    for (size_t i = 0; i < dist.node.degrees.size(); ++i) {
      tmp_dist.node.degrees.push_back(dist.node.degrees.at(i));
      tmp_dist.node.probabilities.push_back(x.at(counter));
      ++counter;
    }
    tmp_dist.renormalize();
    assert(tmp_dist.is_valid());
    return tmp_dist;
  };
  auto dist_to_vec =
      [](const DegreeDistributionPair& dist) -> std::vector<double> {
    // Handles conversion from degree distribution pair to vector.
    std::vector<double> x;
    assert(dist.is_valid());
    for (size_t i = 0; i < dist.check.degrees.size(); ++i) {
      x.push_back(dist.check.probabilities.at(i));
    }
    for (size_t i = 0; i < dist.node.degrees.size(); ++i) {
      x.push_back(dist.node.probabilities.at(i));
    }
    return x;
  };
  double best_obj = std::numeric_limits<double>::max();
  std::mutex best_obj_mutex;
  auto obj_wrapper = [&](const std::vector<double>& x) -> double {
    DegreeDistributionPair tmp_dist = vec_to_dist(x);
    double rate = tmp_dist.rate();
    if (rate < min_rate) {
      return min_rate - rate;
    }
    double obj_val = obj(tmp_dist);
    {
      std::lock_guard<std::mutex> lock(best_obj_mutex);
      std::cout << obj_val;
      if (obj_val < best_obj) {
        std::cout << " new best. Has rate " << rate;
        write(tmp_dist);
        best_obj = obj_val;
      }
      std::cout << std::endl;
    }
    return obj_val;
  };

  bool use_de = true;
  std::vector<double> local_minima;
  if (use_de) {
    using boost::math::optimization::differential_evolution;
    using boost::math::optimization::differential_evolution_parameters;
    auto de_params = differential_evolution_parameters<std::vector<double>>();
    const size_t dimension =
        dist.check.degrees.size() + dist.node.degrees.size();
    de_params.NP = population_size;
    de_params.threads = num_threads;
    de_params.mutation_factor = mutation_factor;
    de_params.crossover_probability = crossover_probability;
    // DimensionlessReal mutation_factor = static_cast<DimensionlessReal>(0.65);
    // DimensionlessReal crossover_probability =
    // static_cast<DimensionlessReal>(0.5); Search on [0, 1]^dimension:
    de_params.lower_bounds.resize(dimension,
                                  std::numeric_limits<double>::min());
    de_params.upper_bounds.resize(dimension, 1);
    // This is a challenging function, increase the max generations 10x from
    // default so we don't terminate prematurely:
    de_params.max_generations *= 10;
    std::vector<double> initial_x = dist_to_vec(dist);
    de_params.initial_guess = &initial_x;
    double value_to_reach = -1.5;
    local_minima =
        differential_evolution(obj_wrapper, de_params, eng, value_to_reach);
  } else {
    using boost::math::optimization::random_search;
    using boost::math::optimization::random_search_parameters;
    auto rs_params = random_search_parameters<std::vector<double>>();
    rs_params.threads = num_threads;
    const size_t dimension =
        dist.check.degrees.size() + dist.node.degrees.size();
    // Search on [0, 1]^dimension:
    rs_params.lower_bounds.resize(dimension, 0);
    rs_params.upper_bounds.resize(dimension, 1);
    std::vector<double> initial_x = dist_to_vec(dist);
    rs_params.initial_guess = &initial_x;
    double value_to_reach = -1.1;
    local_minima = random_search(obj_wrapper, rs_params, eng, value_to_reach);
  }
  std::cout << "Minima: {";
  for (auto l : local_minima) {
    std::cout << l << ", ";
  }
  std::cout << "}\n";
  std::cout << "Value of cost function at minima: " << obj_wrapper(local_minima)
            << "\n";
}
