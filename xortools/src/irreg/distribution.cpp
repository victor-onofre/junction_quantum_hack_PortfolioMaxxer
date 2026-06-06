#include "distribution.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "utils.hpp"

std::string DegreeDistribution::str() {
  std::stringstream ss;
  for (size_t i = 0; i < degrees.size(); ++i) {
    if (probabilities[i] == 0) continue;
    if (i > 0) {
      ss << " + ";
    }
    ss << probabilities[i] << " x^" << (degrees[i] - 1);
  }
  return ss.str();
}

bool DegreeDistribution::is_valid() const {
  return degrees.size() == probabilities.size();
}

bool DegreeDistributionPair::is_valid() const {
  return node.is_valid() and check.is_valid();
}

void DegreeDistributionPair::assert_valid() const {
  if (!is_valid()) {
    throw std::invalid_argument("Invalid degree distribution");
  }
}

std::string DegreeDistribution::degfile_str() const {
  std::stringstream ss;
  for (size_t i = 0; i < degrees.size(); ++i) {
    if (i > 0) ss << " ";
    ss << degrees[i];
  }
  ss << "\n";
  for (size_t i = 0; i < degrees.size(); ++i) {
    if (i > 0) ss << " ";
    ss << std::setprecision(15) << probabilities[i];
  }
  ss << "\n";
  return ss.str();
}

std::string DegreeDistributionPair::degfile_str() const {
  std::stringstream ss;
  ss << node.degfile_str();
  ss << check.degfile_str();
  return ss.str();
}

DegreeDistribution DegreeDistribution::vertex_perspective() const {
  double normalization = 0;
  for (size_t i = 0; i < degrees.size(); ++i) {
    normalization += probabilities[i] / degrees[i];
  }
  DegreeDistribution dist;
  dist.degrees = degrees;
  dist.probabilities.resize(degrees.size());
  for (size_t i = 0; i < degrees.size(); ++i) {
    dist.probabilities[i] = probabilities[i] / degrees[i] / normalization;
  }
  return dist;
}

DegreeDistributionPair DegreeDistributionPair::vertex_perspective() const {
  return DegreeDistributionPair{node.vertex_perspective(),
                                check.vertex_perspective()};
}

DegreeDistributionPair load_degfile(std::string fname) {
  std::ifstream infile(fname);
  if (!infile.is_open()) {
    throw std::runtime_error("Unable to open " + fname);
  }

  std::string line;
  std::vector<std::string> tokens;
  DegreeDistributionPair dist;
  size_t count = 0;
  while (getline(infile, line)) {
    if (line[0] == '#') {
      // skip comment lines
      continue;
    }
    tokens = tokenize(line, ' ');
    for (const auto& token : tokens) {
      switch (count) {
        case 0:
          dist.node.degrees.push_back(std::stoull(token));
          break;
        case 1:
          dist.node.probabilities.push_back(std::stod(token));
          assert(std::stod(token) <= 1);
          break;
        case 2:
          dist.check.degrees.push_back(std::stoull(token));
          break;
        case 3:
          dist.check.probabilities.push_back(std::stod(token));
          assert(std::stod(token) <= 1);
          break;
      }
    }
    ++count;
  }
  dist.assert_valid();
  dist.renormalize();
  return dist;
}

inline void renormalize_vec(std::vector<double>& vec) {
  double sum = 0;
  for (const double& elem : vec) sum += elem;
  for (double& elem : vec) elem /= sum;
  if (sum == 0)
    throw std::invalid_argument(
        "Invalid degree distribution (zero probability "
        "mass before renormalization)");
}

void DegreeDistribution::renormalize() { renormalize_vec(probabilities); }

void DegreeDistributionPair::renormalize() {
  node.renormalize();
  check.renormalize();
}

double DegreeDistribution::mean() const {
  double result = 0;
  for (size_t i = 0; i < degrees.size(); ++i) {
    result += degrees[i] * probabilities[i];
  }
  return result;
}

double DegreeDistributionPair::rate() const {
  DegreeDistributionPair vdist = vertex_perspective();
  double k_mean = vdist.node.mean();
  double D_mean = vdist.check.mean();
  return 1 - k_mean / D_mean;
}
