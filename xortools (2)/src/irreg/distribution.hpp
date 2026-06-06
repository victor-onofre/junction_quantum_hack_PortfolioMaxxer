#ifndef IRREG_DISTRIBUTION_HPP
#define IRREG_DISTRIBUTION_HPP

#include <string>
#include <vector>

struct DegreeDistribution {
  std::vector<size_t> degrees;
  std::vector<double> probabilities;
  std::string str();
  std::string degfile_str() const;
  DegreeDistribution vertex_perspective() const;
  void renormalize();
  double mean() const;
  bool is_valid() const;
};

struct DegreeDistributionPair {
  DegreeDistribution node;
  DegreeDistribution check;
  DegreeDistributionPair vertex_perspective() const;
  std::string degfile_str() const;
  void renormalize();
  double rate() const;
  bool is_valid() const;
  void assert_valid() const;
};

DegreeDistributionPair load_degfile(std::string fname);

#endif  // IRREG_DISTRIBUTION_HPP
