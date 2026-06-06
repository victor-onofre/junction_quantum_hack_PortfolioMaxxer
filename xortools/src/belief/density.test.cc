#include "density.hpp"

#include <gtest/gtest.h>

#include <cassert>
#include <random>

TEST(Density, AssociativeProduct) {
  for (bool fractional_quantization : {true, false}) {
    // Setup
    std::vector<double> quantized_llrs;
    size_t bins = 201;
    double min_llr = -10;
    double max_llr = 10;
    linspace(quantized_llrs, min_llr, max_llr, bins);
    auto [xor_cache, prod_cache] = get_quantized_tables(
        quantized_llrs, min_llr, max_llr, bins, fractional_quantization);

    // Compute some random densities
    std::vector<double> density_a, density_b, density_c;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for (size_t i = 0; i < bins; ++i) {
      density_a.push_back(dist(gen));
      density_b.push_back(dist(gen));
      density_c.push_back(dist(gen));
    }
    renormalize(density_a);
    renormalize(density_b);
    renormalize(density_c);

    // Intermediate density variables
    // Check associativity of check updates
    std::vector<double> foo(bins, 0);
    std::vector<double> bar(bins, 0);
    std::vector<double> baz(bins, 0);
    fast_op(density_a, density_b, foo, xor_cache, bins);
    fast_op(foo, density_c, bar, xor_cache, bins);
    std::vector<double> ab_c = bar;
    zfill(foo);
    zfill(bar);

    fast_op(density_b, density_c, foo, xor_cache, bins);
    fast_op(density_a, foo, bar, xor_cache, bins);
    std::vector<double> a_bc = bar;
    zfill(foo);
    zfill(bar);

    ASSERT_TRUE(is_close(ab_c, a_bc, 1e-3));

    // Check associativity of node updates
    fast_op(density_a, density_b, foo, prod_cache, bins);
    fast_op(foo, density_c, bar, prod_cache, bins);
    ab_c = bar;
    zfill(foo);
    zfill(bar);

    fast_op(density_b, density_c, foo, prod_cache, bins);
    fast_op(density_a, foo, bar, prod_cache, bins);
    a_bc = bar;
    zfill(foo);
    zfill(bar);

    ASSERT_TRUE(is_close(ab_c, a_bc, 1e-2));

    // Test linearity of node updates
    // Compute a (b+c)
    double probability = 0.8;
    for (size_t i = 0; i < bins; ++i) {
      foo[i] += probability * density_b[i];
      foo[i] += (1 - probability) * density_c[i];
    }
    fast_op(density_a, foo, bar, prod_cache, bins);
    zfill(foo);
    // Compute ab + ac
    std::vector<double> ab(bins, 0);
    std::vector<double> ac(bins, 0);
    fast_op(density_a, density_b, ab, prod_cache, bins);
    fast_op(density_a, density_c, ac, prod_cache, bins);
    for (size_t i = 0; i < bins; ++i) {
      foo[i] += probability * ab[i];
      foo[i] += (1 - probability) * ac[i];
    }
    ASSERT_TRUE(is_close(foo, bar, 1e-10));
    zfill(foo);
    zfill(bar);
  }
}

TEST(Density, Rounding) {
  std::vector<double> quantized_llrs = {-3, -2, -1, 0, 1, 2, 3};
  size_t bins = 7;
  double min_llr = -3;
  double max_llr = 3;
  {
    QuantizeResult qr = quantize(0.3, min_llr, max_llr, bins, quantized_llrs,
                                 /*fractional=*/false);
    ASSERT_EQ(qr.i1, 3);
    ASSERT_EQ(qr.p1, 1);
    ASSERT_EQ(qr.i2, 3);
    ASSERT_EQ(qr.p2, 0);
  }
  {
    QuantizeResult qr = quantize(-4, min_llr, max_llr, bins, quantized_llrs,
                                 /*fractional=*/false);
    ASSERT_EQ(qr.i1, 0);
    ASSERT_EQ(qr.p1, 1);
    ASSERT_EQ(qr.i2, 0);
    ASSERT_EQ(qr.p2, 0);
  }
  {
    QuantizeResult qr = quantize(-3, min_llr, max_llr, bins, quantized_llrs,
                                 /*fractional=*/false);
    ASSERT_EQ(qr.i1, 0);
    ASSERT_EQ(qr.p1, 1);
    ASSERT_EQ(qr.i2, 0);
    ASSERT_EQ(qr.p2, 0);
  }
  {
    QuantizeResult qr = quantize(3, min_llr, max_llr, bins, quantized_llrs,
                                 /*fractional=*/false);
    ASSERT_EQ(qr.i1, 6);
    ASSERT_EQ(qr.p1, 1);
    ASSERT_EQ(qr.i2, 6);
    ASSERT_EQ(qr.p2, 0);
  }
  {
    QuantizeResult qr = quantize(3, min_llr, max_llr, bins, quantized_llrs,
                                 /*fractional=*/true);
    ASSERT_EQ(qr.i1, 6);
    ASSERT_EQ(qr.p1, 1);
    ASSERT_EQ(qr.i2, 6);
    ASSERT_EQ(qr.p2, 0);
  }
  {
    QuantizeResult qr = quantize(-4, min_llr, max_llr, bins, quantized_llrs,
                                 /*fractional=*/true);
    ASSERT_EQ(qr.i1, 0);
    ASSERT_EQ(qr.p1, 1);
    ASSERT_EQ(qr.i2, 0);
    ASSERT_EQ(qr.p2, 0);
  }
  {
    QuantizeResult qr = quantize(-2.1, min_llr, max_llr, bins, quantized_llrs,
                                 /*fractional=*/true);
    ASSERT_EQ(qr.i1, 0);
    ASSERT_NEAR(qr.p1, 0.1, 1e-9);
    ASSERT_EQ(qr.i2, 1);
    ASSERT_NEAR(qr.p2, 0.9, 1e-9);
  }
  {
    QuantizeResult qr = quantize(4, min_llr, max_llr, bins, quantized_llrs,
                                 /*fractional=*/true);
    ASSERT_EQ(qr.i1, 6);
    ASSERT_NEAR(qr.p1, 1, 1e-9);
    ASSERT_EQ(qr.i2, 6);
    ASSERT_NEAR(qr.p2, 0, 1e-9);
  }
  {
    QuantizeResult qr = quantize(3, min_llr, max_llr, bins, quantized_llrs,
                                 /*fractional=*/true);
    ASSERT_EQ(qr.i1, 6);
    ASSERT_NEAR(qr.p1, 1, 1e-9);
    ASSERT_EQ(qr.i2, 6);
    ASSERT_NEAR(qr.p2, 0, 1e-9);
  }
}

TEST(DensityEvolution, ContextInvariance) {
  // Check that a (4, 40)-regular LDPC code has the same threshold regardless of
  // the zero-probability entries in the degree distribution.
  double threshold_reg;
  {
    DegreeDistributionPair dist;
    dist.node.degrees.push_back(4);
    dist.node.probabilities.push_back(1.0);
    dist.check.degrees.push_back(40);
    dist.check.probabilities.push_back(1.0);
    auto [lo, hi] = binary_search_threshold(
        dist, 1e-6, 0.5, 1e-6, 100, 100, /*fractional_quantization=*/false,
        /*early_stop=*/true, /*quantization_fiddling=*/false,
        /*shrink_factor=*/0.01, /*verbose=*/false, /*halt_if_leq=*/0);
    threshold_reg = lo;
  }
  double threshold_irreg;
  {
    DegreeDistributionPair dist;
    for (size_t d = 2; d < 101; ++d) {
      dist.node.degrees.push_back(d);
      dist.node.probabilities.push_back(double(d == 4));
      dist.check.degrees.push_back(d);
      dist.check.probabilities.push_back(double(d == 40));
    }
    auto [lo, hi] = binary_search_threshold(
        dist, 1e-6, 0.5, 1e-6, 100, 100, /*fractional_quantization=*/false,
        /*early_stop=*/true, /*quantization_fiddling=*/false,
        /*shrink_factor=*/0.01, /*verbose=*/false, /*halt_if_leq=*/0);
    threshold_irreg = lo;
  }
  std::cout << "threshold_reg = " << threshold_reg
            << " threshold_irreg = " << threshold_irreg << "\n";
  EXPECT_NEAR(threshold_reg, threshold_irreg, 1e-5);
}

TEST(DensityEvolution, BitFlipAndErasure) {}
