#include "sparse/sparse_matrix.hpp"

#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

#include "utils.hpp"

sparse::SparseMatrix::SparseMatrix(size_t nrows, size_t ncols)
    : nrows(nrows), ncols(ncols) {
  row_entries.resize(nrows);
  row_entries_vec.resize(nrows);
  col_entries.resize(ncols);
  col_entries_vec.resize(ncols);
}
void sparse::SparseMatrix::resize(size_t _nrows, size_t _ncols) {
  nrows = _nrows;
  ncols = _ncols;
  row_entries.resize(nrows);
  row_entries_vec.resize(nrows);
  col_entries.resize(ncols);
  col_entries_vec.resize(ncols);
}
void sparse::SparseMatrix::add_entry(sparse::SparseMatrix::Entry entry) {
  if (entry.val == 0)
    throw std::invalid_argument(
        "It does not make sense to add an empty entry to a sparse matrix. What "
        "are you doing?");
  size_t index = entries.size();
  row_entries[entry.row][entry.col] = index;
  row_entries_vec[entry.row].push_back(entry.col);
  col_entries[entry.col][entry.row] = index;
  col_entries_vec[entry.col].push_back(entry.row);
  entries.push_back(entry);
}

void sparse::write_tsv(const sparse::SparseMatrix& H, std::string fname,
                       const std::vector<int>& target) {
  std::ofstream outfile;
  outfile.open(fname);
  if (!outfile.is_open()) {
    throw std::runtime_error("Unable to create output file");
  }
  outfile << H.nrows << "\t" << H.ncols << "\n";
  for (size_t i = 0; i < H.ncols; ++i) {
    if (target[i]) {
      outfile << "0.5000";
    } else {
      outfile << "-0.5000";
    }
    for (auto& [j, index] : H.col_entries[i]) {
      outfile << "\t" << j;
    }
    outfile << "\n";
  }
}

void sparse::write_tsv(const sparse::SparseMatrix& H, std::string fname,
                       std::function<unsigned(void)> gen) {
  std::ofstream outfile;
  outfile.open(fname);
  if (!outfile.is_open()) {
    throw std::runtime_error("Unable to create output file");
  }
  outfile << H.nrows << "\t" << H.ncols << "\n";
  for (size_t i = 0; i < H.ncols; ++i) {
    if (gen() % 2) {
      outfile << "0.5000";
    } else {
      outfile << "-0.5000";
    }
    for (auto& [j, index] : H.col_entries[i]) {
      outfile << "\t" << j;
    }
    outfile << "\n";
  }
}

void sparse::read_tsv(SparseMatrix& H, std::vector<int>& targets,
                      std::string fname) {
  std::ifstream infile(fname);
  if (!infile.is_open()) {
    throw std::runtime_error("Unable to open " + fname);
  }

  std::string line;
  std::vector<std::string> tokens;
  bool first = true;
  size_t col = 0;
  while (getline(infile, line)) {
    if (line[0] == '#') {
      // skip comment lines
      continue;
    }
    tokens = tokenize(line, '\t');
    if (first) {
      first = false;
      if (tokens.size() != 2) {
        throw std::runtime_error("First line of " + fname +
                                 " is misformatted.");
      }
      size_t n = stoi(tokens[0]);
      size_t m = stoi(tokens[1]);
      H.resize(n, m);
      continue;
    }

    // This is a new clause i.e. node
    if (tokens[0][0] == '-') {
      targets.push_back(0);
    } else {
      targets.push_back(1);
    }
    for (size_t v = 1; v < tokens.size(); ++v) {
      sparse::SparseMatrix::Entry entry;
      entry.col = col;
      entry.row = std::stoi(tokens[v]);
      entry.val = 1;
      H.add_entry(entry);
    }
    ++col;
  }
}

void sparse::read_tsv(sparse::SparseMatrix& H, std::string fname) {
  std::ifstream infile(fname);
  if (!infile.is_open()) {
    throw std::runtime_error("Unable to open " + fname);
  }

  std::string line;
  std::vector<std::string> tokens;
  bool first = true;
  size_t col = 0;
  while (getline(infile, line)) {
    if (line[0] == '#') {
      // skip comment lines
      continue;
    }
    tokens = tokenize(line, '\t');
    if (first) {
      first = false;
      if (tokens.size() != 2) {
        throw std::runtime_error("First line of " + fname +
                                 " is misformatted.");
      }
      size_t n = stoi(tokens[0]);
      size_t m = stoi(tokens[1]);
      H.resize(n, m);
      continue;
    }

    // This is a new clause i.e. node
    for (size_t v = 1; v < tokens.size(); ++v) {
      sparse::SparseMatrix::Entry entry;
      entry.col = col;
      entry.row = std::stoi(tokens[v]);
      entry.val = 1;
      H.add_entry(entry);
    }
    ++col;
  }
}

void sparse::read_csv(sparse::SparseMatrix& H,
                      std::vector<std::vector<int>>& targets,
                      std::string fname) {
  std::ifstream infile(fname);
  std::string line;
  std::vector<std::string> tokens;
  bool first = true;
  size_t clause_index = 0;
  while (getline(infile, line)) {
    if (line[0] == '#') {
      // skip comment lines
      continue;
    }
    tokens = tokenize(line, ',');
    if (first) {
      first = false;
      // First line is n, m, p
      if (tokens.size() != 3) {
        throw std::runtime_error("First line of " + fname +
                                 " is misformatted.");
      }
      size_t n = std::stoi(tokens[0]);
      size_t m = std::stoi(tokens[1]);
      size_t p = std::stoi(tokens[2]);
      H.resize(n, m);
      H.modulus = p;
      targets.resize(m);
      continue;
    }
    // Each line thereafter is of the form
    // allowed_elem1,allowed_elem2,...,allowed_elemL,coordinate:coeff
    for (const auto& token : tokens) {
      if (contains(token, ':')) {
        std::vector<std::string> pair = tokenize(token, ':');
        if (pair.size() != 2) {
          throw std::runtime_error("Error: misformatted pair in " + fname);
        }
        int loc = stoi(pair[0]);
        int val = stoi(pair[1]);
        if (loc < 0 || loc > (int)H.nrows) {
          throw std::runtime_error("Error: location out of range in " + fname);
        }
        if (val < 0 || val >= (int)H.modulus) {
          throw std::runtime_error("Error: value out of range in " + fname);
        }
        sparse::SparseMatrix::Entry entry;
        entry.col = clause_index;
        entry.row = loc;
        entry.val = val;
        H.add_entry(entry);
      } else {
        int ibval = stoi(token);
        if (ibval < 0 || ibval > (int)H.modulus) {
          throw std::runtime_error(
              "Invalid RHS " + std::to_string(ibval) + " in clause " +
              std::to_string(clause_index) + " modulus is " +
              std::to_string(H.modulus) + " token: " + token);
        }
        targets[clause_index].push_back((uint16_t)ibval);
      }
    }
    ++clause_index;
  }
}

void sparse::write_csv(SparseMatrix& H, std::vector<std::vector<int>>& targets,
                       std::string fname) {
  std::ofstream outfile(fname);
  if (!outfile.is_open()) {
    throw std::runtime_error("Unable to create output file");
  }

  // Write the first line with nrows, ncols, and modulus
  outfile << H.nrows << "," << H.ncols << "," << H.modulus << "\n";

  // Write each clause
  for (size_t col = 0; col < H.ncols; ++col) {
    std::vector<int> allowed_elements = targets[col];

    // Write allowed elements
    for (size_t i = 0; i < allowed_elements.size(); ++i) {
      if (i > 0) {
        outfile << ",";
      }
      outfile << allowed_elements[i];
    }

    // Write coordinate:value pairs
    for (auto& row : H.col_entries[col]) {
      outfile << ",";
      outfile << row.first << ":" << H.entries[row.second].val;
    }

    outfile << "\n";
  }
}
