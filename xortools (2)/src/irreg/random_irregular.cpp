#include <cassert>

#include "argparse/argparse.hpp"
#include "belief/density.hpp"
#include "irreg/irreg.hpp"
#include "sparse/sparse_matrix.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("random_irregular");
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
  program.add_argument("--seed")
      .help(
          "Random seed (for reproducibility). If blank, uses a hardware-random "
          "seed.")
      .default_value(uint64_t(0))
      .scan<'u', uint64_t>();
  program.add_argument("num-nodes")
      .help("Number of nodes to add")
      .scan<'i', int>();
  program.add_argument("tsv-file-name").help("File to write to");
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

  bool verbose = program.get<bool>("verbose");

  int m = program.get<int>("num-nodes");

  std::string deg_fname = program.get<std::string>("--degfile");
  size_t k = program.get<size_t>("--k");
  size_t D = program.get<size_t>("--D");
  if ((k or D) == (deg_fname.size() > 0)) {
    throw std::invalid_argument(
        "must either supply --k INT --D INT for regular codes, or use "
        "--degfile FILE.deg to specify the degree distribution.");
  }
  // We require vertex perspective degree distributions for sampling irregular
  // codes.
  double design_rate;
  DegreeDistributionPair dist;
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
    dist = dist.vertex_perspective();
  }

  int n = std::max(1, (int)std::round((1 - design_rate) * m));
  std::string fname = program.get<std::string>("tsv-file-name");
  std::cout << "m = " << m << " n = " << n << " fname = " << fname
            << " verbose = " << verbose << "\n";
  if (m < n) {
    throw std::invalid_argument(
        "Got fewer nodes than checks (m=" + std::to_string(m) +
        ", n=" + std::to_string(n) + ")");
  }

  uint64_t seed = program.get<uint64_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::cout << "using seed " << seed << "\n";
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<unsigned> uniform;
  auto gen = [&]() { return uniform(eng); };

  sparse::SparseMatrix H = sample_irreg(m, n, dist, gen, verbose);
  sparse::write_tsv(H, fname, gen);
}
