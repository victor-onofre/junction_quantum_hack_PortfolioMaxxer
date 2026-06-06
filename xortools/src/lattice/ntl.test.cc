#include <gtest/gtest.h>

#include "bleichenbacher_nguyen.hpp"

TEST(NTL, LatticeIntersectionBasisGCD) {
  NTL::mat_ZZ A, B, C;
  A.SetDims(1, 1);
  B.SetDims(1, 1);
  A[0][0] = 6;
  B[0][0] = 9;
  lattice_intersection_basis(A, B, C);
  ASSERT_EQ(C.NumRows(), 1);
  ASSERT_EQ(C.NumCols(), 1);
  // 18 = lcm(6, 9)
  ASSERT_EQ(NTL::conv<int>(C[0][0]), 18);
}

TEST(NTL, LatticeIntersectionBasis) {
  NTL::mat_ZZ A, B, C;
  A.SetDims(1, 2);
  B.SetDims(1, 2);
  A[0][0] = 2;
  A[0][1] = 3;
  B[0][0] = 6;
  B[0][1] = 4;
  lattice_intersection_basis(A, B, C);
  // Intersection(span[2, 3], span[6, 4]) = {[0, 0]}
  ASSERT_EQ(C.NumRows(), 0);
  ASSERT_EQ(C.NumCols(), 2);
}

TEST(NTL, LatticeIntersectionBasisComplicated) {
  NTL::mat_ZZ A, B, C;
  A.SetDims(2, 3);
  B.SetDims(3, 3);
  A[0][0] = 2;
  A[0][1] = 1;
  A[0][2] = 1;
  A[1][0] = 1;
  A[1][1] = 2;
  A[1][2] = 1;

  B[0][0] = 2;
  B[0][1] = 2;
  B[0][2] = 1;
  B[1][0] = 3;
  B[1][1] = 1;
  B[1][2] = 1;
  B[2][0] = 99;
  B[2][1] = 99;
  B[2][2] = 99;
  lattice_intersection_basis(A, B, C);
  // I'm not sure this is correct I just wanted to hit more edge cases of shape
  std::cout << "C = " << C << "\n";
  ASSERT_EQ(C.NumRows(), 2);
  ASSERT_EQ(C.NumCols(), 3);
}

TEST(NTL, LatticeIntersectionBasisComplicatedSwap) {
  NTL::mat_ZZ A, B, C;
  A.SetDims(2, 3);
  B.SetDims(3, 3);
  A[0][0] = 2;
  A[0][1] = 1;
  A[0][2] = 1;
  A[1][0] = 1;
  A[1][1] = 2;
  A[1][2] = 1;

  B[0][0] = 2;
  B[0][1] = 2;
  B[0][2] = 1;
  B[1][0] = 3;
  B[1][1] = 1;
  B[1][2] = 1;
  B[2][0] = 99;
  B[2][1] = 99;
  B[2][2] = 99;
  std::swap(A, B);
  lattice_intersection_basis(A, B, C);
  // I'm not sure this is correct I just wanted to hit more edge cases of shape
  std::cout << "C = " << C << "\n";
  ASSERT_EQ(C.NumRows(), 2);
  ASSERT_EQ(C.NumCols(), 3);
}

TEST(NTL, LatticeIntersectionBasisIdentity) {
  NTL::mat_ZZ A, B, C;
  A.SetDims(2, 3);
  B.SetDims(3, 3);
  A[0][0] = 2;
  A[0][1] = 3;
  A[0][2] = 4;
  A[1][0] = 9;
  A[1][1] = 10;
  A[1][2] = 11;

  B[0][0] = 1;
  B[0][1] = 0;
  B[0][2] = 0;
  B[1][0] = 0;
  B[1][1] = 1;
  B[1][2] = 0;
  B[2][0] = 0;
  B[2][1] = 0;
  B[2][2] = 1;
  std::swap(A, B);
  lattice_intersection_basis(A, B, C);
  // I'm not sure this is correct I just wanted to hit more edge cases of shape
  std::cout << "C = " << C << "\n";
  ASSERT_EQ(C.NumRows(), 2);
  ASSERT_EQ(C.NumCols(), 3);
}
