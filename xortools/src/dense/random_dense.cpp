#include <cassert>

#include "argparse/argparse.hpp"
#include "sparse/sparse_matrix.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("random_irregular");
  program.add_argument("--seed")
      .help(
          "Random seed (for reproducibility). If blank, uses a hardware-random "
          "seed.")
      .default_value(uint64_t(0))
      .scan<'u', uint64_t>();
  program.add_argument("num-nodes-clauses")
      .help("Number of nodes (= number of clauses) to add")
      .scan<'i', int>();
  program.add_argument("num-checks-variables")
      .help("Number of checks (= number of variables) to add")
      .scan<'i', int>();
  program.add_argument("modulus").help("modulus (e.g. 2)").scan<'i', int>();
  program.add_argument("list-size")
      .help("number of acceptable symbols for each coordinate")
      .scan<'i', int>();
  program.add_argument("csv-file-name").help("File to write to");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  int modulus = program.get<int>("modulus");
  int m = program.get<int>("num-nodes-clauses");
  int n = program.get<int>("num-checks-variables");
  int list_size = program.get<int>("list-size");
  std::string fname = program.get<std::string>("csv-file-name");

  uint64_t seed = program.get<uint64_t>("--seed");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  std::cout << "using seed " << seed << "\n";
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<unsigned> uniform;
  auto gen = [&]() { return uniform(eng); };
  sparse::SparseMatrix B(n, m);
  B.modulus = modulus;
  std::vector<std::vector<int>> targets(m);
  std::vector<int> values(modulus);
  std::iota(values.begin(), values.end(), 0);
  for (int i = 0; i < m; ++i) {
    std::shuffle(values.begin(), values.end(), eng);
    targets[i].insert(targets[i].end(), values.begin(),
                      values.begin() + list_size);
    std::sort(targets[i].begin(), targets[i].end());
    assert(targets[i].size() == (size_t)list_size);

    for (int j = 0; j < n; ++j) {
      int coeff = gen() % modulus;
      if (coeff == 0) continue;
      sparse::SparseMatrix::Entry entry;
      entry.row = j;
      entry.col = i;
      entry.val = coeff;
      B.add_entry(entry);
    }
  }
  sparse::write_csv(B, targets, fname);
}
