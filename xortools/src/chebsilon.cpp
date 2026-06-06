#include <gmp.h>
#include <math.h>
#include <string.h>

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "deps/gmp-6.3.0/gmp.h"
#include "deps/cxxopts/cxxopts.hpp"
#include "utils.hpp"
#include "xorsat.hpp"


struct GmpRational {
  mpq_t value;

  GmpRational() { mpq_init(value); }

  GmpRational(const GmpRational& other) {
    mpq_init(value);
    mpq_set(value, other.value);
  }

  GmpRational(unsigned val) {
    mpq_init(value);
    mpq_set_ui(value, val, 1);
  }

  GmpRational(int val) {
    mpq_init(value);
    mpq_set_si(value, val, 1);
  }

  GmpRational(double val) {
    mpq_init(value);
    mpq_set_d(value, val);
  }

  bool operator==(const GmpRational& other) const {
    return mpq_equal(value, other.value);
  }

  GmpRational& operator=(const GmpRational& other) {
    if (this != &other) {
      mpq_set(value, other.value);
    }
    return *this;
  }

  GmpRational& operator=(int num) {
    mpq_set_si(value, num, 1);  // Set the rational number as num/1
    return *this;
  }

  ~GmpRational() { mpq_clear(value); }

  double to_double() const { return mpq_get_d(value); }

  std::string str() const {
    char* c_str = mpq_get_str(nullptr, 10, value);
    std::string result(c_str);
    // Free the C string allocated by mpq_get_str
    // void (*freefunc)(void *, size_t);
    // mp_get_memory_functions(nullptr, nullptr, &freefunc);
    free(c_str);
    return result;
  }

  // Operator overloading
  GmpRational operator+(const GmpRational& rhs) const {
    GmpRational result;
    mpq_add(result.value, value, rhs.value);
    return result;
  }

  GmpRational& operator+=(const GmpRational& rhs) {
    mpq_add(value, value, rhs.value);
    return *this;
  }

  GmpRational& operator/=(const GmpRational& other) {
    mpq_div(value, value, other.value);
    return *this;
  }

  GmpRational operator/(const GmpRational& other) const {
    GmpRational result;
    mpq_div(result.value, value, other.value);
    return result;
  }

  GmpRational& operator/=(int num) {
    mpq_t temp;
    mpq_init(temp);
    mpq_set_si(temp, num, 1);  // Set temp as num/1
    mpq_div(value, value, temp);
    mpq_clear(temp);
    return *this;
  }

  GmpRational operator*(const GmpRational& rhs) const {
    GmpRational result;
    mpq_mul(result.value, value, rhs.value);
    return result;
  }

  GmpRational& operator*=(const GmpRational& rhs) {
    mpq_mul(value, value, rhs.value);
    return *this;
  }

  GmpRational operator-(const GmpRational& rhs) const {
    GmpRational result;
    mpq_sub(result.value, value, rhs.value);
    return result;
  }

  GmpRational& operator-=(const GmpRational& rhs) {
    mpq_sub(value, value, rhs.value);
    return *this;
  }
};

struct Polynomial {
  std::vector<GmpRational> coeffs;

  // Add a coefficient
  void add_coefficient(const GmpRational& coeff) { coeffs.push_back(coeff); }

  // Evaluate the polynomial at a point x
  GmpRational evaluate_at(const GmpRational& x) const {
    GmpRational result(0), x_pow(1);

    result += coeffs[0];
    for (size_t i=1; i<coeffs.size(); ++i) {
      x_pow *= x;  // Update x_pow for the next term
      result += coeffs[i] * x_pow;
    }

    return result;
  }

  Polynomial operator+(const Polynomial& rhs) const {
    Polynomial result;
    size_t max_size = std::max(coeffs.size(), rhs.coeffs.size());
    result.coeffs.resize(max_size);

    for (size_t i = 0; i < max_size; ++i) {
      if (i < coeffs.size()) result.coeffs[i] += coeffs[i];
      if (i < rhs.coeffs.size()) result.coeffs[i] += rhs.coeffs[i];
    }

    return result;
  }

  Polynomial operator-(const Polynomial& other) const {
    Polynomial result;
    size_t max_size = std::max(coeffs.size(), other.coeffs.size());
    result.coeffs.resize(max_size);

    for (size_t i = 0; i < max_size; ++i) {
      if (i < coeffs.size()) {
        result.coeffs[i] += coeffs[i];
      }
      if (i < other.coeffs.size()) {
        result.coeffs[i] -= other.coeffs[i];
      }
    }

    return result;
  }

  Polynomial& operator+=(const Polynomial& rhs) {
    size_t max_size = std::max(coeffs.size(), rhs.coeffs.size());
    coeffs.resize(max_size);

    for (size_t i = 0; i < rhs.coeffs.size(); ++i) {
      coeffs[i] += rhs.coeffs[i];
    }

    return *this;
  }

  Polynomial operator*(const Polynomial& other) const {
    Polynomial result;
    // Initializing result coefficients with zero values
    for (size_t i = 0; i < coeffs.size() + other.coeffs.size() - 1; ++i) {
      result.coeffs.push_back(GmpRational(0));
    }

    // Polynomial multiplication
    for (size_t i = 0; i < coeffs.size(); ++i) {
      for (size_t j = 0; j < other.coeffs.size(); ++j) {
        result.coeffs[i + j] += coeffs[i] * other.coeffs[j];
      }
    }

    return result;
  }

  Polynomial& operator*=(const GmpRational& scalar) {
    for (auto& coeff : coeffs) {
      coeff *= scalar;
    }
    return *this;
  }

  std::string str() const {
    std::stringstream ss;

    bool firstTerm = true;
    for (size_t i = 0; i < coeffs.size(); ++i) {
      // Skip terms with a coefficient of zero
      if (coeffs[i] == GmpRational(0)) continue;

      // Add plus sign for non-first terms
      if (!firstTerm) {
        ss << " + ";
      } else {
        firstTerm = false;
      }

      // Add the coefficient
      ss << coeffs[i].str();  // Assuming GmpRational has a str() method

      // Add the variable part
      if (i > 0) {
        ss << "x";
        if (i > 1) {
          ss << "^" << i;
        }
      }
    }

    // Handle the case of all zero coefficients
    if (firstTerm) {
      return "0";
    }

    return ss.str();
  }
};

Polynomial operator*(const Polynomial& poly, const GmpRational& scalar) {
  Polynomial result = poly;
  for (auto& coeff : result.coeffs) {
    coeff *= scalar;
  }
  return result;
}

Polynomial operator*(const GmpRational& scalar, const Polynomial& poly) {
  return poly * scalar;
}

std::vector<Polynomial> chebyshev_T(unsigned n) {
  Polynomial T0, T1, Tnext;

  // Initialize T0(x) = 1
  GmpRational one;
  one = 1;
  T0.add_coefficient(one);

  std::vector<Polynomial> polys;
  polys.push_back(T0);
  if (n == 0) return polys;

  // Initialize T1(x) = x
  T1.add_coefficient(GmpRational(0));
  T1.add_coefficient(GmpRational(1));
  Polynomial Px = T1;

  polys.push_back(T1);
  if (n == 1) return polys;

  for (unsigned i = 1; i < n; ++i) {
    Tnext = Px * T1 * GmpRational(2);  // 2xT_n(x)
    Tnext = Tnext - T0;                // - T_{n-1}(x)

    T0 = T1;
    T1 = Tnext;
    polys.push_back(T1);
  }

  return polys;
}

void test_chebyshev_T() {
  // Define the expected coefficients for Chebyshev polynomials T0 to T9
  std::vector<std::vector<int>> expected_coeffs = {
      {1},                                      // T0
      {0, 1},                                   // T1
      {-1, 0, 2},                               // T2
      {0, -3, 0, 4},                            // T3
      {1, 0, -8, 0, 8},                         // T4
      {0, 5, 0, -20, 0, 16},                    // T5
      {-1, 0, 18, 0, -48, 0, 32},               // T6
      {0, -7, 0, 56, 0, -112, 0, 64},           // T7
      {1, 0, -32, 0, 160, 0, -256, 0, 128},     // T8
      {0, 9, 0, -120, 0, 432, 0, -576, 0, 256}  // T9
  };

  std::vector<Polynomial> T = chebyshev_T(9);
  for (unsigned n = 0; n <= 9; ++n) {
    const std::vector<int>& expected = expected_coeffs[n];

    // Check if the size of the coefficients matches
    std::cout << "n = " << n << "\n";
    std::cout << T[n].str() << "\n";
    assert(T[n].coeffs.size() == expected.size());

    // Check each coefficient
    for (size_t i = 0; i < expected.size(); ++i) {
      assert(T[n].coeffs[i] == GmpRational(expected[i]));
    }
  }

  std::cout << "All tests passed." << std::endl;
}

void gmp_demo() {
  mpq_t frac, result;
  mpq_init(frac);    // Initialize the rational number
  mpq_init(result);  // Initialize the result rational number

  mpq_set_ui(frac, 1, 2);    // Set frac to 1/2
  mpq_set_ui(result, 1, 2);  // Set result to 1

  // Raise frac to the 1000th power
  unsigned exponent = 1000;
  for (unsigned i = 0; i < exponent; ++i) {
    mpq_mul(result, frac, result);
  }

  // Output the result
  char* str = mpq_get_str(NULL, 10, result);
  std::cout << "1/2 to the 1000th power is: " << str << std::endl;

  // Clean up
  mpq_clear(frac);
  mpq_clear(result);
  free(str);
}

// Fills up probabilities[i] with the probability of obtaining a solution
// satisfying i (out of m) constraints. p = prime (usually this is 2)
void chebyshev_stats(unsigned m, unsigned ell, unsigned p) {
  std::vector<GmpRational> magnitudes(m + 1);
  GmpRational one_over_p = GmpRational(1) / GmpRational(p);
  GmpRational one_minus_one_over_p = GmpRational(p - 1) / GmpRational(p);
  mpz_t binom;
  mpz_init(binom);
  // Compute f(t) as the sum of the first l chebyshev polynomials, with an
  // appropriate offset
  std::vector<Polynomial> T = chebyshev_T(ell);
  Polynomial h;
  for (unsigned i = 0; i <= ell; ++i) {
    GmpRational coeff(1);
    if (i % 2) coeff = -1;
    h += coeff * T[i];
  }
  h.coeffs[0] = 0;
  GmpRational c(1);
  c *= unsigned(sqrt(double(ell) * double(m) / 7.0) * 1e5);
  c /= 100000;
  auto f = [&](unsigned t) -> GmpRational {
    GmpRational x = (GmpRational(m) / GmpRational(p) - GmpRational(t)) / c;
    return h.evaluate_at(x);
  };
  GmpRational sum_magnitudes(0);
  // binomial_prob is equal to (1/p)^t * (1-1/p)^(m-t) * (m choose t)
  GmpRational binomial_prob = 1;
  // When t = 0, this is binomial_prob = (1-1/p)^m
  for (unsigned i=0; i<m; ++i) binomial_prob *= one_minus_one_over_p;
  for (unsigned t = 0; t <= m; ++t) {
    // Set the magnitude to be the binomial coefficient (m choose t)
    magnitudes[t] = binomial_prob;
    // The recurrence relation is
    // binomial_prob(t+1) = binomial_prob(t)/(p - 1)*(m - t)/(t + 1)
    binomial_prob *= (m-t);
    binomial_prob /= (p-1);
    binomial_prob /= (t+1);
    // Multiply by f(t)^2
    GmpRational f_t = f(t);
    magnitudes[t] *= f_t * f_t;

    sum_magnitudes += magnitudes[t];
  }
  for (unsigned t = 0; t <= m; ++t) {
    magnitudes[t] /= sum_magnitudes;
    double prob = magnitudes[t].to_double();
    if (prob != 0) {
      std::cout << "t = " << t << " probability = " << prob << "\n";
    }
  }
  GmpRational prob_at_least_t(0);
  for (int t = m; t >= 0; --t) {
    prob_at_least_t += magnitudes[t];
    double prob = prob_at_least_t.to_double();
    if (prob != 0) {
      std::cout << "t >= " << t << " probability = " << prob << "\n";
    }
  }
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("chebsilon", "estimate the probabilities that DQI gives an instance satisfying t (or more) clauses out of the m clauses total.");
  options.add_options()
  ("p", "prime to work modulo", cxxopts::value<int>()->default_value("2"))
  ("m", "number of constraints", cxxopts::value<int>())
  ("l,ell", "number of constraints that can be "
  "successfully decoded", cxxopts::value<int>())
  ("h,help", "Print usage")
    ;

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  int p = result["p"].as<int>();
  int m = result["m"].as<int>();
  int l = result["l"].as<int>();

  gmp_demo();
  test_chebyshev_T();

  chebyshev_stats(m, l, p);
}
