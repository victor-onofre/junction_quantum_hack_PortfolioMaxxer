#ifndef _GERRYMANDERING_PROBLEM_DEF_H
#define _GERRYMANDERING_PROBLEM_DEF_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "probability.h"

struct GerrymanderingInstance {
  int m;
  int b;
  int budget;
  int target;
  probability_distribution_t p_s;

  friend std::ostream& operator<<(std::ostream& os,
                                  const GerrymanderingInstance& gi) {
    os << "{m=" << gi.m << ", b=" << gi.b << ", budget=" << gi.budget
       << ", target=" << gi.target << ", p(s)=" << gi.p_s << "}";
    return os;
  }

  void write(std::ostream& os) {
    os << "m: " << m << "\n";
    os << "b: " << b << "\n";
    os << "budget: " << budget << "\n";
    os << "target: " << target << "\n";
    os << "p(s): " << p_s << std::endl;
  }

  friend std::istream& operator>>(std::istream& is,
                                  GerrymanderingInstance& gi) {
    std::string ignore;
    is >> ignore >> gi.m;
    is >> ignore >> gi.b;
    is >> ignore >> gi.budget;
    is >> ignore >> gi.target;
    char c;
    is >> ignore >> c;
    gi.p_s.resize(gi.b + 1);
    for (auto& p : gi.p_s) is >> p >> c;
    return is;
  }

  void write(const char* fp) {
    std::ofstream out(fp);
    write(out);
    out.close();
  }

  static GerrymanderingInstance from_file(const char* fp) {
    std::ifstream in(fp);
    GerrymanderingInstance gi;
    in >> gi;
    in.close();
    return gi;
  }

  bool operator==(const GerrymanderingInstance& other) const {
    if (!(m == other.m && b == other.b && budget == other.b &&
          target == other.target))
      return false;
    for (int i = 0; i <= b; i++) {
      if (std::abs(p_s[i] - other.p_s[i]) > 1e-12) {
        return false;
      }
    }
    return true;
  }

  template <typename random_number_engine_t>
  static GerrymanderingInstance make_random(int max_m, int max_b,
                                            int max_budget, int max_target,
                                            random_number_engine_t& rne) {
    int m = std::uniform_int_distribution<int>(1, max_m)(rne);
    int b = std::uniform_int_distribution<int>(1, max_b)(rne);
    int budget = std::uniform_int_distribution<int>(0, max_budget)(rne);
    int target = std::uniform_int_distribution<int>(
        0, std::min(max_target + 0LL, m * (long long)b))(rne);
    probability_distribution_t p_s(b + 1);
    std::uniform_real_distribution<probability_t> u(0, 1);
    for (int i = 0; i <= b; i++) {
      p_s[i] = u(rne);
    }
    std::sort(p_s.begin(), p_s.end());
    return GerrymanderingInstance({m, b, budget, target, p_s});
  }
};

struct MaximizationResult {
  probability_t phi_c;
  std::vector<int> choices;

  bool operator<(const MaximizationResult& other) const {
    return phi_c < other.phi_c;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const MaximizationResult& mr) {
    os << "MaximizationResult{" << mr.phi_c << ", " << mr.choices << "}";
    return os;
  }

  void swap(MaximizationResult& other) {
    std::swap(phi_c, other.phi_c);
    std::swap(choices, other.choices);
  }

  static MaximizationResult constant(int n, probability_t p = 0, int s = 0) {
    return MaximizationResult{p, std::vector<int>(n, s)};
  }

  void swap_if_better(MaximizationResult& other) {
    if (*this < other) {
      this->swap(other);
    }
  }
};

#endif
