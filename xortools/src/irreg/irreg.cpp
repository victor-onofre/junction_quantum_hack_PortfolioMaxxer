#include "irreg/irreg.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <queue>
#include <random>
#include <set>

using namespace sparse;

void print_degree_distribution(const std::map<size_t, size_t>& deg_counts) {
  std::cout << "d_i = ";
  for (auto it : deg_counts) {
    std::cout << it.first << ", ";
  }
  std::cout << "\n";
  std::cout << "n_i = ";
  for (auto it : deg_counts) {
    std::cout << it.second << ", ";
  }
  std::cout << "\n";
}

// Given an out_degs vector preallocated with the appropriate length, fills in
// with degrees approximately according to the given distribution.
void assign_degs(const std::vector<size_t>& dist_degs,
                 const std::vector<double>& dist_probs,
                 std::vector<size_t>& out_degs) {
  double cum_prob = dist_probs[0];
  size_t d = 0;
  for (size_t i = 0; i < out_degs.size(); ++i) {
    if (i > cum_prob * out_degs.size()) {
      ++d;
      cum_prob += dist_probs.at(d);
    }
    out_degs[i] = dist_degs.at(d);
  }
}

// Adds i to the q degs[i] times, in a random order
void shuffle_push(const std::vector<size_t>& degs, std::queue<size_t>& q,
                  std::function<unsigned(void)> gen) {
  std::vector<size_t> elements;
  for (size_t i = 0; i < degs.size(); ++i) {
    for (size_t v = 0; v < degs[i]; ++v) {
      elements.push_back(i);
    }
  }
  unsigned seed = gen();
  std::mt19937 g(seed);
  std::shuffle(elements.begin(), elements.end(), g);
  for (size_t i : elements) {
    q.push(i);
  }
}

// Adds i to the q degs[i] times, such that the first edge is highest priority,
// and later edges are lower and lower priority
void fair_push(const std::vector<size_t>& degs, std::queue<size_t>& q,
               std::function<unsigned(void)> gen) {
  std::vector<std::vector<size_t>> edges;
  for (size_t i = 0; i < degs.size(); ++i) {
    for (size_t v = 0; v < degs[i]; ++v) {
      if (v >= edges.size()) edges.resize(v + 1);
      edges[v].push_back(i);
    }
  }

  // Randomize the ordering within each level
  unsigned seed = gen();
  std::mt19937 g(seed);
  for (size_t v = 0; v < edges.size(); ++v) {
    std::shuffle(edges[v].begin(), edges[v].end(), g);
    for (size_t i : edges[v]) {
      q.push(i);
    }
  }
}

SparseMatrix sample_irreg(int m, int n, const DegreeDistributionPair& dist,
                          std::function<unsigned(void)> gen, bool verbose) {
  SparseMatrix H(n, m);

  // Assign node degrees
  std::vector<size_t> node_degs(m);
  assign_degs(dist.node.degrees, dist.node.probabilities, node_degs);
  for (size_t i = 0; i < node_degs.size(); ++i) {
    assert(node_degs[i] >= 2);
  }

  // Assign check degrees
  std::vector<size_t> check_degs(n);
  assign_degs(dist.check.degrees, dist.check.probabilities, check_degs);

  // Fill up queues with node and check indices
  std::queue<size_t> node_q, check_q;
  fair_push(node_degs, node_q, gen);
  shuffle_push(check_degs, check_q, gen);

  // Exhaust the queues.
  size_t num_edges = 0;
  while (!node_q.empty() and !check_q.empty()) {
    size_t i = node_q.front();
    node_q.pop();
    size_t j = check_q.front();
    check_q.pop();
    // Get a j that is not already connected to i
    size_t num_pops = 1;
    while ((num_pops < check_q.size() + 1 and H.col_entries[i].count(j))) {
      check_q.push(j);
      j = check_q.front();
      check_q.pop();
      ++num_pops;
    }
    // We never want to add a redundant edge. So we just don't ever do that.
    // Even if it is the only possibility left. We just skip adding this edge
    // and move on.
    if (H.col_entries[i].count(j)) {
      // There is no point re-pushing on the node side since we know it has no
      // avaialable check edges to match to.
      check_q.push(j);
      continue;
    }
    SparseMatrix::Entry entry;
    entry.col = i;
    entry.row = j;
    entry.val = 1;
    H.add_entry(entry);
    ++num_edges;
  }
  if (verbose) {
    std::cout << "leftover node_q.size() = " << node_q.size()
              << " check_q.size() = " << check_q.size() << "\n";
  }
  double k_eff = double(num_edges) / double(H.ncols);
  double D_eff = double(num_edges) / double(H.nrows);
  std::map<size_t, size_t> node_deg_counts, check_deg_counts;
  for (size_t i = 0; i < H.ncols; ++i) {
    ++node_deg_counts[H.col_entries[i].size()];
  }
  for (size_t j = 0; j < H.nrows; ++j) {
    ++check_deg_counts[H.row_entries[j].size()];
  }
  if (verbose) {
    std::cout << "edges " << num_edges << " k_eff " << k_eff << " D_eff "
              << D_eff << "\n";
    std::cout << "empirical node degree distribution (vertex perspective):\n";
    print_degree_distribution(node_deg_counts);
    std::cout << "empirical check degree distribution (vertex perspective):\n";
    print_degree_distribution(check_deg_counts);
  }

  return H;
}
