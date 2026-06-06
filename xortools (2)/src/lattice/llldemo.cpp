#include <string>

#include "NTL/LLL.h"
#include "NTL/ZZ.h"
#include "gmpwrap.hpp"

// Bazel build command: bazel build src:llldemo && ./bazel-bin/src/llldemo

int main() {
  NTL::ZZ a, b, c;
  a = 101;
  b = 1001;
  c = a * b;
  std::cout << "a = " << a << " b = " << b << " c = " << c << "\n";

  gmpwrap::GmpRational foo(3);
  gmpwrap::GmpRational bar(6);

  std::cout << "foo = " << foo.str() << " bar = " << bar.str()
            << " foo / bar = " << (foo / bar).str() << "\n";

  NTL::mat_ZZ B;
  B.SetDims(2, 2);
  B[0][0] = 11;
  B[0][1] = 14;
  B[1][0] = 2;
  B[1][1] = 5;
  std::cout << "original matrix B =\n" << B << "\n";
  NTL::ZZ det2;
  long res = NTL::LLL(det2, B, /*verbose=*/10);
  std::cout << "reduced matrix B (LLL-reduced) =\n" << B << "\n";
  std::cout << "det(lattice)^2 = " << det2 << "\n";
  std::cout << "got rank = " << res << "\n";
}
