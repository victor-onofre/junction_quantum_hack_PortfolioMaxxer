#ifndef LATTICE_GMPWRAP_H
#define LATTICE_GMPWRAP_H

#include <gmp.h>
#include <gmpxx.h>

#include <string>

namespace gmpwrap {

struct GmpRational {
  mpq_t value;

  GmpRational();

  GmpRational(const GmpRational& other);

  GmpRational(unsigned val);

  GmpRational(int val);

  GmpRational(double val);

  bool operator==(const GmpRational& other) const;

  GmpRational& operator=(const GmpRational& other);

  GmpRational& operator=(int num);

  ~GmpRational();

  double to_double() const;

  std::string str() const;

  // Operator overloading
  GmpRational operator+(const GmpRational& rhs) const;

  GmpRational& operator+=(const GmpRational& rhs);

  GmpRational& operator/=(const GmpRational& other);

  GmpRational operator/(const GmpRational& other) const;

  GmpRational& operator/=(int num);
  GmpRational operator*(const GmpRational& rhs) const;

  GmpRational& operator*=(const GmpRational& rhs);

  GmpRational operator-(const GmpRational& rhs) const;

  GmpRational& operator-=(const GmpRational& rhs);

  GmpRational operator-() const;
};

}  // namespace gmpwrap

#endif  // LATTICE_GMPWRAP_H
