#include "gmpwrap.hpp"

using namespace gmpwrap;

GmpRational::GmpRational() { mpq_init(value); }

GmpRational::GmpRational(const GmpRational& other) {
  mpq_init(value);
  mpq_set(value, other.value);
}

GmpRational::GmpRational(unsigned val) {
  mpq_init(value);
  mpq_set_ui(value, val, 1);
}

GmpRational::GmpRational(int val) {
  mpq_init(value);
  mpq_set_si(value, val, 1);
}

GmpRational::GmpRational(double val) {
  mpq_init(value);
  mpq_set_d(value, val);
}

bool GmpRational::operator==(const GmpRational& other) const {
  return mpq_equal(value, other.value);
}

GmpRational& GmpRational::operator=(const GmpRational& other) {
  if (this != &other) {
    mpq_set(value, other.value);
  }
  return *this;
}

GmpRational& GmpRational::operator=(int num) {
  mpq_set_si(value, num, 1);  // Set the rational number as num/1
  return *this;
}

GmpRational::~GmpRational() { mpq_clear(value); }

double GmpRational::to_double() const { return mpq_get_d(value); }

std::string GmpRational::str() const {
  char* c_str = mpq_get_str(nullptr, 10, value);
  std::string result(c_str);
  // Free the C string allocated by mpq_get_str
  free(c_str);
  return result;
}

// Operator overloading
GmpRational GmpRational::operator+(const GmpRational& rhs) const {
  GmpRational result;
  mpq_add(result.value, value, rhs.value);
  return result;
}

GmpRational& GmpRational::operator+=(const GmpRational& rhs) {
  mpq_add(value, value, rhs.value);
  return *this;
}

GmpRational& GmpRational::operator/=(const GmpRational& other) {
  mpq_div(value, value, other.value);
  return *this;
}

GmpRational GmpRational::operator/(const GmpRational& other) const {
  GmpRational result;
  mpq_div(result.value, value, other.value);
  return result;
}

GmpRational& GmpRational::operator/=(int num) {
  mpq_t temp;
  mpq_init(temp);
  mpq_set_si(temp, num, 1);  // Set temp as num/1
  mpq_div(value, value, temp);
  mpq_clear(temp);
  return *this;
}

GmpRational GmpRational::operator*(const GmpRational& rhs) const {
  GmpRational result;
  mpq_mul(result.value, value, rhs.value);
  return result;
}

GmpRational& GmpRational::operator*=(const GmpRational& rhs) {
  mpq_mul(value, value, rhs.value);
  return *this;
}

GmpRational GmpRational::operator-(const GmpRational& rhs) const {
  GmpRational result;
  mpq_sub(result.value, value, rhs.value);
  return result;
}

GmpRational& GmpRational::operator-=(const GmpRational& rhs) {
  mpq_sub(value, value, rhs.value);
  return *this;
}

GmpRational GmpRational::operator-() const {
  GmpRational result(*this);
  mpq_neg(result.value, this->value);
  return result;
}
