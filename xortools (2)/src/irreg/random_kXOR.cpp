#include <cassert>

#include "argparse/argparse.hpp"
#include "belief/density.hpp"
#include "irreg/irreg.hpp"
#include "sparse/sparse_matrix.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("random_kXOR");
  program.add_argument("--k")
      .help("number of edges per node")
      .default_value(size_t(0))
      .scan<'u', size_t>();
  program.add_argument("--n-over-m")
      .help("the ratio n / m")
      .default_value(double(0.5))
      .scan<'g', double>();
  program.add_argument("--seed")
      .help(
          "Random seed (for reproducibility). If blank, uses a hardware-random "
          "seed.")
      .default_value(uint64_t(0))
      .scan<'u', uint64_t>();
  program.add_argument("num-nodes")
      .help("Number of nodes to add")
      .scan<'u', size_t>();
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
  size_t m = program.get<size_t>("num-nodes");
  double n_over_m = program.get<double>("--n-over-m");
  size_t k = program.get<size_t>("--k");
  if (k == 0) {
    throw std::invalid_argument("Must provide k > 0");
  }

  std::string fname = program.get<std::string>("tsv-file-name");

  uint64_t seed = program.get<uint64_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::cout << "using seed " << seed << "\n";
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<unsigned> uniform;
  auto gen = [&]() { return uniform(eng); };

  int n = std::max(1, int(std::round(n_over_m * double(m))));
  if (verbose) {
    std::cout << "m = " << m << " n = " << n << std::endl;
  }

  sparse::SparseMatrix H(n, m);

  std::vector<size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, ..., n-1
  for (size_t i = 0; i < m; ++i) {
    std::vector<size_t> random_subset;
    std::sample(indices.begin(), indices.end(),
                std::back_inserter(random_subset), k, eng);
    assert(random_subset.size() == k);
    for (size_t j : random_subset) {
      sparse::SparseMatrix::Entry entry;
      entry.row = j;
      entry.col = i;
      entry.val = 1;
      H.add_entry(entry);
    }
  }

  sparse::write_tsv(H, fname, gen);
}
