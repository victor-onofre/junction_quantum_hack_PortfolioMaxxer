#include <argparse/argparse.hpp>
#include <sstream>

#include "bruteforce.h"
#include "knapsack.h"
#include "problem_def.h"

int main(int argc, char** argv) {
  argparse::ArgumentParser cli("gerrymandering");
  std::string file_path, solver;
  bool fast = false;
  cli.add_argument("--file_path")
      .required()
      .store_into(file_path)
      .help(
          "path to a file containing a problem description (see example.txt)");
  cli.add_argument("--solver")
      .store_into(solver)
      .default_value("backtracking")
      .help("solver. one of {knapsack, backtracking}");
  cli.add_argument("--fast").store_into(fast).default_value(false).flag().help(
      "if used with the knapsack solver will use the fast (less accurate) "
      "version of the solver");
  try {
    cli.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << cli;
    std::exit(1);
  }

  GerrymanderingInstance gi =
      GerrymanderingInstance::from_file(file_path.c_str());
  std::cout << gi << std::endl;
  if (solver == "knapsack") {
    std::cout << "knapsack: " << knapsack::maximize_probability(gi, fast)
              << std::endl;
  } else {
    std::cout << "bruteforce: " << bruteforce::maximize_probability(gi)
              << std::endl;
  }
  return 0;
}
