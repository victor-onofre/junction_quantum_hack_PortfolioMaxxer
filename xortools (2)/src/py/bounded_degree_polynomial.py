import galois
import numpy as np
import itertools
import math
from collections import defaultdict, Counter

class BoundedDegreePolynomial:
  @staticmethod
  def generate_canonical_monomial_keys(n, D):
    """Generate sorted tuples representing all monomials of degree 0 to D."""
    for deg in range(0, D + 1):
      for combination in itertools.combinations_with_replacement(range(n), deg):
        yield combination

  @staticmethod
  def num_canonical_monomial_keys(n, D):
    """Return the number of monomials of degree 0 to D in n variables."""
    return sum(math.comb(n + d - 1, d) for d in range(0, D + 1))

  def __init__(self, n, D, p):
    self.n = n
    self.D = D
    self.GF = galois.GF(p)
    self.coeffs = {}  # Sparse representation: only store nonzero terms

  def set(self, monomial, coeff):
    k = tuple(sorted(monomial))
    val = self.GF(coeff)
    if val == 0:
      self.coeffs.pop(k, None)
    else:
      self.coeffs[k] = val

  def get(self, monomial):
    return self.coeffs.get(tuple(sorted(monomial)), self.GF(0))

  @classmethod
  def constant_value(cls, value, n=0, D=0, p=2):
    instance = cls(n, D, p)
    instance.set((), value)
    return instance

  @staticmethod
  def format_monomial(monomial, sub_super=True, x='x', mul_sign=''):
    subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    superscript_digits = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    if not monomial:
      return ""
    var_counts = Counter(monomial)
    parts = []
    for i in sorted(var_counts):
      sub_i = f"{x}{i}"
      if sub_super:
        sub_i = sub_i.translate(subscript_digits)
      exp = var_counts[i]
      if exp == 1:
        parts.append(sub_i)
      else:
        str_exp = str(exp)
        if sub_super:
          str_exp = str_exp.translate(superscript_digits)
        parts.append(sub_i + str_exp)
    return mul_sign.join(parts)

  def m4gb_str(self, x='X'):
    terms = []
    for monomial, coeff in self.coeffs.items():
      monomial_str = self.format_monomial(monomial, sub_super=False, x=x, mul_sign='*')
      if monomial_str == "":
        term_str = f"{coeff}"
      elif coeff == 1:
        term_str = monomial_str
      else:
        term_str = f'{coeff}*{monomial_str}'
      terms.append(term_str)
    return ' + '.join(terms) if terms else '0'

  def __str__(self):
    terms = []
    for monomial, coeff in self.coeffs.items():
      monomial_str = self.format_monomial(monomial)
      if monomial_str == "":
        term_str = f"{coeff}"
      elif coeff == 1:
        term_str = monomial_str
      else:
        term_str = f"{coeff}*{monomial_str}"
      terms.append(term_str)
      if len(terms) > 10:
        return " + ".join(terms[:10]) + " + ..."
    return " + ".join(terms) if terms else "0"

  def __repr__(self):
    return f"BoundedDegreePolynomial(n={self.n}, D={self.D}, p={int(self.GF.characteristic)}, terms={len(self.coeffs)})"

  def __add__(self, other):
    if isinstance(other, BoundedDegreePolynomial):
      assert self.n == other.n and self.D == other.D and self.GF == other.GF
      result = BoundedDegreePolynomial(self.n, self.D, int(self.GF.characteristic))
      keys = set(self.coeffs) | set(other.coeffs)
      for k in keys:
        val = self.get(k) + other.get(k)
        if val != 0:
          result.coeffs[k] = val
      return result
    else:  # scalar + polynomial
      return self + BoundedDegreePolynomial.constant_value(other, self.n, self.D, int(self.GF.characteristic))

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    assert self.n == other.n and self.D == other.D and self.GF == other.GF
    result = BoundedDegreePolynomial(self.n, self.D, int(self.GF.characteristic))
    keys = set(self.coeffs) | set(other.coeffs)
    for k in keys:
      val = self.get(k) - other.get(k)
      if val != 0:
        result.coeffs[k] = val
    return result

  def __mul__(self, other):
    if isinstance(other, BoundedDegreePolynomial):
      assert self.n == other.n and self.D == other.D and self.GF == other.GF
      result_coeffs = defaultdict(lambda: self.GF(0))
      for m1, c1 in self.coeffs.items():
        for m2, c2 in other.coeffs.items():
          m_combined = tuple(sorted(m1 + m2))
          if len(m_combined) > self.D:
            raise ValueError(f"Multiplication exceeds max degree D={self.D}: {m1} * {m2} = {m_combined}")
          result_coeffs[m_combined] += c1 * c2
      result = BoundedDegreePolynomial(self.n, self.D, int(self.GF.characteristic))
      for k, v in result_coeffs.items():
        if v != 0:
          result.coeffs[k] = v
      return result
    else:  # Scalar multiplication
      scalar = self.GF(other)
      result = BoundedDegreePolynomial(self.n, self.D, int(self.GF.characteristic))
      for k, v in self.coeffs.items():
        product = scalar * v
        if product != 0:
          result.coeffs[k] = product
      return result

  def __rmul__(self, other):
    return self * other

  def __sub__(self, other):
    if isinstance(other, BoundedDegreePolynomial):
      assert self.n == other.n and self.D == other.D and self.GF == other.GF
      result = BoundedDegreePolynomial(self.n, self.D, int(self.GF.characteristic))
      keys = set(self.coeffs) | set(other.coeffs)
      for k in keys:
        val = self.get(k) - other.get(k)
        if val != 0:
          result.coeffs[k] = val
      return result
    else:  # Scalar subtraction: self - c
      return self - BoundedDegreePolynomial.constant_value(other, self.n, self.D, int(self.GF.characteristic))

  def __rsub__(self, other):
    # Scalar minus polynomial: c - self
    return BoundedDegreePolynomial.constant_value(other, self.n, self.D, int(self.GF.characteristic)) - self

  def __truediv__(self, other):
    # Scalar division
    scalar = self.GF(other)
    if scalar == 0:
      raise ZeroDivisionError("Division by zero scalar")
    result = BoundedDegreePolynomial(self.n, self.D, int(self.GF.characteristic))
    for k, v in self.coeffs.items():
      q = v / scalar
      if q != 0:
        result.coeffs[k] = q
    return result
