#include <cassert>
#include <filesystem>
#include <set>
#include <thread>

#include "argparse/argparse.hpp"
#include "optimize/anneal.hpp"
#include "optimize/problem.hpp"
#include "sparse/sparse_matrix.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("strip_high_degree_clauses");
  program.add_argument("instance").help("Instance in .tsv format");

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
  program.add_argument("--seed")
      .help(
          "Random seed (for reproducibility). If blank or 0, uses a "
          "hardware-random "
          "seed.")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--write")
      .help(
          "Write stripped problems as .tsv files in the current working "
          "directory.")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--verbose")
      .help("increase output verbosity")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  bool verbose = program.get<bool>("--verbose");
  bool do_write_files = program.get<bool>("--write");
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
  const double min_beta = program.get<double>("--min-beta");
  const double max_beta = program.get<double>("--max-beta");
  std::cout << "min_beta = " << min_beta << " max_beta = " << max_beta
            << " iterations = " << iterations << "\n";
  double min_acceptance_rate = program.get<double>("--min-acceptance-rate");

  std::string fname = program.get<std::string>("instance");

  std::vector<int> target;
  sparse::SparseMatrix H(0, 0);
  sparse::read_tsv(H, target, fname);
  std::filesystem::path path = std::filesystem::path(fname);
  fname = path.filename();
  std::string directory = path.parent_path();
  std::map<int, MaxXORSATProblem> subproblems;
  MaxXORSATProblem problem{H, target};
  strip_high_degree_clauses(problem, subproblems, verbose);

  std::vector<std::thread> threads;
  size_t num_subproblems = subproblems.size();
  std::vector<int> subproblem_num_sat(num_subproblems);
  std::vector<int> degree_cutoffs(num_subproblems);
  size_t counter = 0;
  for (const auto& [degree, subproblem] : subproblems) {
    degree_cutoffs[counter] = degree;
    uint64_t subthread_seed = gen();
    threads.push_back(
        std::thread([&subproblem_num_sat, &subproblems, &H, &target, degree,
                     do_write_files, min_acceptance_rate, min_beta, max_beta,
                     directory, fname, iterations, subthread_seed, counter]() {
          std::mt19937_64 eng(subthread_seed);
          std::uniform_int_distribution<uint64_t> uniform;
          auto gen = [&]() -> uint64_t { return uniform(eng); };
          auto x =
              anneal(subproblems[degree].H, subproblems[degree].targets,
                     iterations, min_beta, max_beta, min_acceptance_rate, gen,
                     /*verbose=*/false);
          int num_sat_full = evaluate(H, target, x);
          if (do_write_files) {
            std::string stripped_fname = directory + "/stripped_lb_" +
                                         std::to_string(degree) + "_" + fname;
            std::cout << "writing to " << stripped_fname << "\n";
            sparse::write_tsv(subproblems[degree].H, stripped_fname,
                              subproblems[degree].targets);
          }
          subproblem_num_sat[counter] = num_sat_full;
        }));
    ++counter;
  }
  assert(threads.size() == num_subproblems);
  assert(subproblem_num_sat.size() == num_subproblems);
  int best_num_sat_full = 0;
  for (size_t i = 0; i < num_subproblems; ++i) {
    threads[i].join();
    if (verbose) {
      std::cout << "stripped clauses of degree > " << degree_cutoffs[i]
                << " annealed solution satisfying " << subproblem_num_sat[i]
                << " clauses on the full problem\n";
    }
    best_num_sat_full = std::max(best_num_sat_full, subproblem_num_sat[i]);
  }
  std::cout << "best_num_sat_full = " << best_num_sat_full << "\n";
}
