#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

#include "utils.hpp"

namespace sparse {

struct SparseMatrix {
  struct Entry {
    size_t row;
    size_t col;
    int val;
  };
  size_t modulus = 2;
  size_t nrows;
  size_t ncols;
  std::vector<std::map<size_t, size_t>> row_entries;
  std::vector<std::vector<size_t>> row_entries_vec;
  std::vector<std::map<size_t, size_t>> col_entries;
  std::vector<std::vector<size_t>> col_entries_vec;
  std::vector<Entry> entries;
  SparseMatrix() = default;
  SparseMatrix(size_t nrows, size_t ncols);
  void resize(size_t _nrows, size_t _ncols);
  void add_entry(Entry entry);
};

void write_tsv(const SparseMatrix& H, std::string fname,
               std::function<unsigned(void)> gen);

void write_tsv(const SparseMatrix& H, std::string fname,
               const std::vector<int>& target);

void read_tsv(SparseMatrix& H, std::string fname);
void read_tsv(SparseMatrix& H, std::vector<int>& targets, std::string fname);

void read_csv(SparseMatrix& H, std::vector<std::vector<int>>& targets,
              std::string fname);
void write_csv(SparseMatrix& H, std::vector<std::vector<int>>& targets,
               std::string fname);

}  // namespace sparse

#endif  // SPARSE_MATRIX_HPP
