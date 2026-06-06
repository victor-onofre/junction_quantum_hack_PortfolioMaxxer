#include <string.h>

#include <cassert>
#include <fstream>

#include "argparse/argparse.hpp"
#include "belief/density.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("density");
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
  program.add_argument("--shrink-factor")
      .help(
          "Shrinking the bit error rate by this factor is interpreted as "
          "indicating below threshold.")
      .default_value(double(0.5))
      .scan<'g', double>();
  program.add_argument("--density-fname")
      .help(
          "(optional) a file to write the densities to between each iteration")
      .default_value(std::string(""));
  program.add_argument("--delta")
      .help("delta value to run density evolution at")
      .default_value(double(0))
      .scan<'g', double>();
  program.add_argument("--bins")
      .help("Number of LLR bins to use for quantization (must be odd)")
      .default_value(size_t(1001))
      .scan<'u', size_t>();
  program.add_argument("--iterations")
      .help("Number of BP iterations to model (default 1000)")
      .default_value(size_t(1000))
      .scan<'u', size_t>();
  program.add_argument("--verbose")
      .help("increase output verbosity")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--fractional-quantization")
      .help(
          "use noah\'s fractional quantization technique (i.e. stochastic "
          "rounding emulation)")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--quantization-fiddling")
      .help(
          "Fiddle with the min_llr, max_llr, and num_llr_bins so that there is "
          "a bin which exactly represents the initial LLR from the channel "
          "parameters.")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--no-early-stop")
      .help(
          "Disables early-stopping optimization (wherein the density evolution "
          "halts once the bit error rate changes by a factor of <= 1/2 or >= "
          "2)")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--max-delta")
      .help("Maximum error rate to consider for the binary search")
      .default_value(double(0.5))
      .scan<'g', double>();
  program.add_argument("--precision")
      .help(
          "In binary search mode, how close should the upper and lower bounds "
          "be before terminating the binary search.")
      .default_value(double(1e-4))
      .scan<'g', double>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  bool verbose = program.get<bool>("verbose");
  bool early_stop = not program.get<bool>("no-early-stop");
  bool fractional_quantization = program.get<bool>("fractional-quantization");
  bool quantization_fiddling = program.get<bool>("quantization-fiddling");
  double max_delta = program.get<double>("--max-delta");
  std::string deg_fname = program.get<std::string>("--degfile");
  size_t k = program.get<size_t>("--k");
  size_t D = program.get<size_t>("--D");
  if ((k or D) and (!k or !D))
    throw std::invalid_argument("Invalid --k and/or --D provided");
  size_t num_llr_bins = program.get<size_t>("--bins");
  if ((k or D) == (deg_fname.size() > 0)) {
    throw std::invalid_argument(
        "must either supply --k INT --D INT for regular codes, or use "
        "--degfile FILE.deg to specify the degree distribution.");
  }
  double shrink_factor = program.get<double>("--shrink-factor");
  DegreeDistributionPair dist;
  if (k or D) {
    dist.node.degrees.push_back(k);
    dist.node.probabilities.push_back(1.0);
    dist.check.degrees.push_back(D);
    dist.check.probabilities.push_back(1.0);
  } else {
    assert(deg_fname.size());
    dist = load_degfile(deg_fname);
  }
  if (verbose) {
    std::cout << "dist.node.str() = " << dist.node.str() << "\n";
    std::cout << "dist.check.str() = " << dist.check.str() << "\n";
  }

  double eps = 1e-10;
  size_t num_iterations = program.get<size_t>("--iterations");

  std::string density_fname = program.get<std::string>("--density-fname");
  double delta = program.get<double>("--delta");
  if ((delta == 0) != (density_fname.size() == 0)) {
    throw std::invalid_argument(
        "Must either provide --delta ERROR_RATE and --density-fname "
        "FILE.DENSITY OR omit both of these for binary search mode.");
  }
  bool binary_search_mode = (delta == 0);
  double precision = program.get<double>("--precision");
  if (binary_search_mode) {
    // Binary search to find the threshold to within 4 significant digits.
    std::cout << "precision = " << precision << " bins = " << num_llr_bins
              << " num_iterations = " << num_iterations << "\n";
    auto [delta_min, delta_max] = binary_search_threshold(
        dist, /*delta_min=*/eps, /*delta_max=*/max_delta - eps,
        /*precision=*/precision, num_llr_bins, num_iterations,
        fractional_quantization, early_stop, quantization_fiddling,
        shrink_factor, verbose, /*halt_if_leq=*/0.0);
    std::cout << "threshold appears to be at least " << std::setprecision(15)
              << delta_min << " - " << delta_max << "\n";
    return 0;
  }

  // Run at a specific delta and write to file mode
  std::ofstream outfile;
  outfile.open(density_fname);
  if (!outfile.is_open()) {
    throw std::runtime_error("Unable to create output file");
  }
  // Add comments to the density evolution data file
  outfile << "# lambda(x) = " << dist.node.str() << "\n";
  outfile << "# rho(x) = " << dist.check.str() << "\n";
  outfile << "# delta = " << delta << "\n";
  auto callback = [&](const std::vector<double>& vec) {
    outfile << vec[0];
    for (size_t i = 1; i < vec.size(); ++i) {
      outfile << " " << vec[i];
    }
    outfile << "\n";
  };
  bool below = density_evolution(
      dist, delta, num_llr_bins, num_iterations, fractional_quantization,
      early_stop, quantization_fiddling, shrink_factor, verbose, callback);
  if (verbose) {
    std::cout << "got below = " << below << "\n";
  }
  outfile.close();
  return 0;
}
