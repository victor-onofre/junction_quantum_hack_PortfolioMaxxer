import functools
import inspect
import os
from typing import Dict, Set, TYPE_CHECKING, Union

import attrs
import numpy as np
from qualtran import Bloq
from qualtran.bloqs.basic_gates import CNOT, Toffoli
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
    _GF2MulViaKaratsubaImpl,
    BinaryPolynomialMultiplication,
    GF2MulK,
    GF2MulViaKaratsuba,
    GF2ShiftRight,
    GF2MulMBUC,
)
from qualtran.symbolics import is_symbolic

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.resource_counting import BloqCountDictT, BloqCountT
    from qualtran.symbolics import SympySymbolAllocator


def _qubit_name_to_index(name: str, n: int) -> int:
    name = name.replace('_', '')
    return int(name[1:]) + (ord(name[0].lower()) - ord('a')) * n


@attrs.frozen
class PolyMul(BinaryPolynomialMultiplication):
    r"""An optimized version of BinaryPolynomialMultiplication that uses fewer toffolis.

    For now this bloq uses the hard coded circuits for $2 \leq n \leq 6$ in circuits/ directory and
    falls back to BinaryPolynomialMultiplication for other cases.
    """

    @functools.cached_property
    def _optimized_ops(self):
        location = os.path.dirname(inspect.getabsfile(PolyMul))
        path = os.path.join(location, 'circuits', f'poly_mul_{self.n}.txt')
        lines = open(path).read().splitlines()

        return [l.split() for l in lines]

    def build_composite_bloq(
        self, bb: 'BloqBuilder', f: 'SoquetT', g: 'SoquetT', h: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n) or not (2 <= self.n <= 6):
            return super().build_composite_bloq(bb, f, g, h)

        qs = f.tolist() + g.tolist() + h.tolist()
        for op_name, *qubit_names in self._optimized_ops:
            if op_name == 'cnot':
                c, t = [_qubit_name_to_index(name, self.n) for name in qubit_names]
                qs[c], qs[t] = bb.add(CNOT(), ctrl=qs[c], target=qs[t])
            else:
                c1, c2, t = [_qubit_name_to_index(name, self.n) for name in qubit_names]
                (qs[c1], qs[c2]), qs[t] = bb.add(Toffoli(), ctrl=(qs[c1], qs[c2]), target=qs[t])

        f = np.array(qs[: self.n])
        g = np.array(qs[self.n : 2 * self.n])
        h = np.array(qs[2 * self.n :])
        return {'f': f, 'g': g, 'h': h}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if is_symbolic(self.n) or not (2 <= self.n <= 6):
            return super().build_call_graph(ssa)

        cnot = 0
        tof = 0
        for op_name, *_ in self._optimized_ops:
            if op_name == 'cnot':
                cnot += 1
            else:
                tof += 1

        return {Toffoli(): tof, CNOT(): cnot}


@attrs.frozen
class _GF2MulViaOptimizedPolyMul(_GF2MulViaKaratsubaImpl):
    def build_composite_bloq(
        self, bb: 'BloqBuilder', f: 'Soquet', g: 'Soquet', h: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        f_arr = bb.split(f)
        g_arr = bb.split(g)
        h_arr = bb.split(h)
        f_arr = f_arr[::-1]
        g_arr = g_arr[::-1]
        h_arr = h_arr[::-1]

        for i in range(self.n - self.k):
            f_arr[self.k + i], f_arr[i] = bb.add(CNOT(), ctrl=f_arr[self.k + i], target=f_arr[i])
        for i in range(self.n - self.k):
            g_arr[self.k + i], g_arr[i] = bb.add(CNOT(), ctrl=g_arr[self.k + i], target=g_arr[i])
        f_arr[: self.k], g_arr[: self.k], h_arr[: 2 * self.k - 1] = bb.add(
            PolyMul(self.k), f=f_arr[: self.k], g=g_arr[: self.k], h=h_arr[: 2 * self.k - 1]
        )

        for i in range(self.n - self.k):
            g_arr[self.k + i], g_arr[i] = bb.add(CNOT(), ctrl=g_arr[self.k + i], target=g_arr[i])
        for i in range(self.n - self.k):
            f_arr[self.k + i], f_arr[i] = bb.add(CNOT(), ctrl=f_arr[self.k + i], target=f_arr[i])

        h = bb.join(h_arr[: self.n][::-1], self.qgf)
        h = bb.add(GF2MulK.from_polynomials([0, self.k], self.m_x).adjoint(), g=h)
        h_arr[: self.n] = bb.split(h)[::-1]
        f_arr[self.k : self.n], g_arr[self.k : self.n], h_arr[: 2 * (self.n - self.k) - 1] = bb.add(
            PolyMul(self.n - self.k),
            f=f_arr[self.k : self.n],
            g=g_arr[self.k : self.n],
            h=h_arr[: 2 * (self.n - self.k) - 1],
        )

        h = bb.join(h_arr[: self.n][::-1], self.qgf)
        h = bb.add(GF2ShiftRight(self.qgf, self.k), f=h)
        h_arr[: self.n] = bb.split(h)[::-1]

        f_arr[: self.k], g_arr[: self.k], h_arr[: 2 * self.k - 1] = bb.add(
            PolyMul(self.k), f=f_arr[: self.k], g=g_arr[: self.k], h=h_arr[: 2 * self.k - 1]
        )
        h = bb.join(h_arr[: self.n][::-1], self.qgf)
        h = bb.add(GF2MulK.from_polynomials([0, self.k], self.m_x), g=h)
        h_arr[: self.n] = bb.split(h)[::-1]

        f_arr = f_arr[::-1]
        g_arr = g_arr[::-1]
        h_arr = h_arr[::-1]
        f = bb.join(f_arr, self.qgf)
        g = bb.join(g_arr, self.qgf)
        h = bb.join(h_arr, self.qgf)
        return {'f': f, 'g': g, 'h': h}


@attrs.frozen
class GF2MulViaOptimizedPolyMul(GF2MulViaKaratsuba):
    r"""An optimized version of GF2MulViaKaratsuba that uses fewer toffolis.

    For now this bloq uses the hard coded circuits for $n \leq 12$ in circuits/ directory and
    falls back to GF2MulViaKaratsuba for other cases.
    """

    @functools.cached_property
    def _gf2_mul_impl(self) -> Bloq:
        impl = _GF2MulViaOptimizedPolyMul(self.m_x)
        return impl

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet', **soqs: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n) or self.n > 12:
            return super().build_composite_bloq(bb, x, y)

        result = bb.allocate(self.n, self.qgf)

        x, y, result = bb.add_from(self._gf2_mul_impl, f=x, g=y, h=result)

        return {'x': x, 'y': y, 'result': result}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if is_symbolic(self.n) or self.n > 12:
            return super().build_call_graph(ssa)

        if self.n == 1:
            return {Toffoli(): 1}
        if 2 * self.k == self.n:
            return {
                CNOT(): 4 * (self.n - self.k),
                PolyMul(self.k): 3,
                GF2MulK.from_polynomials([0, self.k], self.m_x): 1,
                GF2MulK.from_polynomials([0, self.k], self.m_x).adjoint(): 1,
                GF2ShiftRight(self.qgf, self.k): 1,
            }
        return {
            CNOT(): 4 * (self.n - self.k),
            PolyMul(self.k): 2,
            PolyMul(self.n - self.k): 1,
            GF2MulK.from_polynomials([0, self.k], self.m_x): 1,
            GF2MulK.from_polynomials([0, self.k], self.m_x).adjoint(): 1,
            GF2ShiftRight(self.qgf, self.k): 1,
        }

    def adjoint(self) -> 'Bloq':
        return GF2MulMBUCOpt(self.qgf)


@attrs.frozen
class GF2MulMBUCOpt(GF2MulMBUC):
    def adjoint(self) -> 'Bloq':
        return GF2MulViaOptimizedPolyMul(self.qgf)
