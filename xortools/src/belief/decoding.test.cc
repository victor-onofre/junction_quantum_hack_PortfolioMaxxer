#include "decoding.hpp"

#include <gtest/gtest.h>

TEST(Decoding, Entropy) {
  ASSERT_NEAR(binary_entropy(0.5), 1.0, 0.01);
  ASSERT_NEAR(binary_entropy(0.11), 0.5, 0.01);
  ASSERT_NEAR(binary_entropy(1), 0, 0.01);
  ASSERT_NEAR(binary_entropy(0), 0, 0.01);
}

TEST(Decoding, InverseEntropy) {
  ASSERT_NEAR(inverse_binary_entropy(0.5), 0.11, 0.01);
  ASSERT_NEAR(inverse_binary_entropy(1), 0.5, 0.001);
}

TEST(Decoding, MaximumEll) { ASSERT_EQ(maximum_possible_ell(50, 100), 12); }
