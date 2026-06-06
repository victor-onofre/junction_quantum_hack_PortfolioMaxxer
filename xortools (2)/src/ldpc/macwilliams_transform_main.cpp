#include <gmpxx.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "argparse/argparse.hpp"
#include "macwilliams_transform.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("krawtchouk");
  program.add_argument("--weight-dist-file")
      .help(
          "Newline-delimited arbitrary precision decimal numbers giving the "
          "weight distribution of the input code.")
      .default_value(std::string(""));

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  std::string weight_dist_fname =
      program.get<std::string>("--weight-dist-file");
  if (!weight_dist_fname.size()) {
    throw std::invalid_argument("Must provide a weight dist");
  }
  std::vector<gmpwrap::GmpRational> code_weight_dist;

  load_weight_dist(weight_dist_fname, code_weight_dist);
  // print_weight_dist(code_weight_dist);

  std::vector<gmpwrap::GmpRational> dual_code_weight_dist;
  constexpr size_t num_worker_threads = 64;
  macwilliams_transform(code_weight_dist, dual_code_weight_dist,
                        num_worker_threads);
  print_weight_dist(dual_code_weight_dist);
  return 0;
}
