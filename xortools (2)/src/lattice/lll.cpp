#include "NTL/LLL.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NTL/ZZ.h"
#include "gmpwrap.hpp"

// Bazel build command: bazel build src:lll && ./bazel-bin/src/lll
// Test case: /var/folders/c2/kdrrtsfs41gfqygr18mf25n00000gn/T/tmpbt9p24qn.tsv
// Test case: /var/folders/c2/kdrrtsfs41gfqygr18mf25n00000gn/T/tmphmaeg312.tsv

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: lattice basis" << std::endl;
    return 0;
  }

  // Load file containing lattice basis
  std::string file_name = argv[1];
  std::ifstream temp_file(file_name);

  if (!temp_file) {
    std::cout << "Unable to open file " << file_name << std::endl;
    return 0;
  }

  // Initialize lattice basis array
  std::vector<std::vector<int>> basis;
  std::vector<int> row;

  // Read each line from the file
  std::string line;
  while (std::getline(temp_file, line)) {
    std::stringstream ss(line);
    std::string item;

    // Tokenize the line based on tabs
    while (std::getline(ss, item, '\t')) {
      row.push_back(std::stoi(item));
    }

    basis.push_back(row);
    row.clear();
  }

  // Close the file
  temp_file.close();

  // Dimensions of lattice basis
  int n = basis.size();
  int m = basis[0].size();

  // Build NTL matrix
  NTL::ZZ det2;
  long verbose;
  NTL::mat_ZZ B;
  B.SetDims(n, m);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      B[i][j] = basis[i][j];
    }
  }

  // Run LLL on B
  NTL::LLL(det2, B, verbose = 0);
  std::cout << B << std::endl;
}
