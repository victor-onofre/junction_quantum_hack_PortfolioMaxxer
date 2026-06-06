from typing import Dict, Union, Set, Mapping, Optional, Tuple

import attrs
from functools import cached_property

from qualtran import Bloq, QGF, Signature, Register, Side, QDType, QBit, QUInt, QAny
from qualtran.symbolics import bit_length, SymbolicInt
from qualtran.resource_counting import SympySymbolAllocator, BloqCountT, BloqCountDictT
from qualtran.bloqs.basic_gates import XGate, Toffoli, TwoBitCSwap
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.arithmetic import AddK, XorK
from qualtran.bloqs.gf_arithmetic import GF2Addition, GF2ShiftLeft
from galois import Poly, GF


def gf2_division_using_eea(a: GF, b: GF) -> GF:
    r"""Compute a // b using EEA"""
    # 1. Compute dialog of `b`.
    gf = type(a)
    d = gf.degree + 1
    poly = GF(2).Zeros(2 * d + 3)  # 2n qubits
    b_poly = Poly(b.vector())
    poly[:d] = gf.irreducible_poly.coefficients(order="asc")
    poly[-b_poly.degree - 1 :] = b_poly.coefficients(order="desc")
    poly = poly.tolist()
    n_f, n_g = d, b_poly.degree + 1
    dialog = []
    # Apply the dialog to `a`
    u, v = a, gf(0)  # 2n qubits
    for it in range(2 * d):
        # 2n iterations
        is_swap = n_f > n_g and poly[-1] != 0  # log(n) Toffoli.
        if is_swap:  # 1 Toffoli
            poly = poly[::-1]  # len(poly) // 2 Toffolis
            u, v = v, u  # n Toffolis
            n_f, n_g = n_g, n_f  # log(n) Toffoli
        dialog.append(poly[-1])
        if dialog[-1]:
            assert n_f <= n_g
            # len(poly) Toffolis total.
            u = u + v  # n Toffolis
            for j in range(len(poly) // 2):
                if j <= n_f:
                    poly[len(poly) - j - 1] ^= poly[j]
        assert poly.pop() == 0
        u = u // gf(2)  # Clifford
        n_g -= 1  # log(n) Toffoli
        if u == 0:
            return dialog, v, it
    return dialog, v, 2 * d
    # print(u, v)
    # assert u == 0
    # return dialog, v # 2n + n ==> 3n qubits.


@attrs.frozen
class ComputeDegree(Bloq):
    dtype: QGF
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [Register('x', dtype=self.dtype), Register('deg', dtype=self.quint, side=side)]
        )

    @cached_property
    def n(self) -> int:
        return self.dtype.bitsize + 1

    @cached_property
    def logn(self) -> int:
        return bit_length(self.n)

    @cached_property
    def quint(self) -> QUInt:
        return QUInt(self.logn)

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        deg = Poly(vals["x"].vector()).degree + 1
        if self.is_adjoint:
            assert vals.pop("deg") == deg
        else:
            vals["deg"] = deg
        return vals

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {Toffoli(): self.n}

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        return f'{cls_name}†' if self.is_adjoint else cls_name


@attrs.frozen
class GreaterThan(Bloq):
    r"""Low ancilla bloq for quantum-quantum comparison"""

    dtype: QUInt
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register('x', dtype=self.dtype),
                Register('y', dtype=self.dtype),
                Register('target', dtype=QBit(), side=side),
            ]
        )

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {Toffoli(): self.dtype.bitsize}

    def on_classical_vals(
        self, x: int, y: int, target: Optional[int] = None
    ) -> Dict[str, 'ClassicalValT']:
        if self.is_adjoint:
            assert target == (x > y)
        return {'x': x, 'y': y} if self.is_adjoint else {'x': x, 'y': y, 'target': x > y}

    def __str__(self) -> str:
        return 'LessThan†' if self.is_adjoint else 'LessThan'


@attrs.frozen
class CSWAP(Bloq):
    dtype: QDType

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(ctrl=QBit(), x=self.dtype, y=self.dtype)

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        if vals["ctrl"] == 1:
            vals["x"], vals["y"] = vals["y"], vals["x"]
        return vals

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {TwoBitCSwap(): self.dtype.num_qubits}

    def adjoint(self) -> 'Bloq':
        return self


@attrs.frozen
class GMinusFPoly(Bloq):
    r"""g -= f in shared poly representation using unary iteration"""

    dtype: QGF
    poly_len: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('poly', dtype=QBit(), shape=(self.poly_len,)),
                Register('n_f', dtype=self.quint),
            ]
        )

    @cached_property
    def n(self) -> int:
        return self.dtype.bitsize + 1

    @cached_property
    def logn(self) -> int:
        return bit_length(self.n)

    @cached_property
    def quint(self) -> QUInt:
        return QUInt(self.logn)

    def on_classical_vals(self, poly: list, n_f: int) -> Dict[str, 'ClassicalValT']:
        if poly[-1]:
            for j in range(1, len(poly) // 2):
                if j - 1 < n_f:
                    poly[len(poly) - j - 1] ^= poly[j]
        return {'poly': poly, 'n_f': n_f}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {Toffoli(): self.poly_len - 2}

    def adjoint(self) -> 'Bloq':
        return self


@attrs.frozen
class GF2DivisionUsingDialog(Bloq):
    """Computes result = x // y"""

    dtype: QGF

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', self.dtype, side=Side.LEFT),
                Register('y', self.dtype, side=Side.LEFT),
                Register('junk', QAny(self.junk_bitsize), side=Side.RIGHT),
                Register('x_by_y', self.dtype, side=Side.RIGHT),
            ]
        )

    @cached_property
    def n(self) -> int:
        return self.dtype.bitsize + 1

    @cached_property
    def logn(self) -> int:
        return bit_length(self.n)

    @cached_property
    def quint(self) -> QUInt:
        return QUInt(self.logn)

    @cached_property
    def junk_bitsize(self) -> SymbolicInt:
        return sum(reg.total_bits() for reg in self.junk_regs)

    @cached_property
    def junk_regs(self) -> Tuple[Register, ...]:
        return (
            Register('dialog', dtype=QBit(), shape=(2 * self.n - 1,), side=Side.RIGHT),
            Register('n_f_gt_ngs', dtype=QBit(), shape=(2 * self.n - 1,), side=Side.RIGHT),
            Register('n_f', dtype=self.quint, side=Side.RIGHT),
            Register('n_g', dtype=self.quint, side=Side.RIGHT),
        )

    @cached_property
    def unpartition_bloq(self) -> Partition:
        return Partition(self.junk_bitsize, self.junk_regs, partition=False)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, x: 'SoquetT', y: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        # Initialize poly for shared-register EEA on (y, irreducible_poly)
        y, n_g = bb.add(ComputeDegree(self.dtype), x=y)
        n_f = bb.allocate(dtype=self.quint)
        n_f = bb.add(XorK(self.quint, self.n), x=n_f)
        poly = [bb.allocate(dtype=QBit()) for _ in range(self.n + 1)]
        for i, c in enumerate(self.dtype.gf_type.irreducible_poly.coefficients(order="asc")):
            if c:
                poly[i] = bb.add(XGate(), q=poly[i])
        poly = [*poly, *bb.split(y)]
        dialog = []
        u, v = x, bb.allocate(dtype=self.dtype)
        # n_f_gt_ng_meas = []
        n_f_gt_ngs = []
        for _ in range(2 * self.n - 1):
            # Perform is_swap and do MBUC for n_f > n_g
            n_f, n_g, n_f_gt_ng = bb.add(GreaterThan(self.quint), x=n_f, y=n_g)
            (n_f_gt_ng, poly[-1]), is_swap = bb.add(And(), ctrl=(n_f_gt_ng, poly[-1]))
            for i in range(len(poly) // 2):
                is_swap, poly[i], poly[-i - 1] = bb.add(
                    TwoBitCSwap(), ctrl=is_swap, x=poly[i], y=poly[-i - 1]
                )
            is_swap, u, v = bb.add(CSWAP(self.dtype), ctrl=is_swap, x=u, y=v)
            is_swap, n_f, n_g = bb.add(CSWAP(self.quint), ctrl=is_swap, x=n_f, y=n_g)
            (n_f_gt_ng, poly[-1]) = bb.add(
                And().adjoint(), ctrl=[n_f_gt_ng, poly[-1]], target=is_swap
            )
            # n_f_gt_ng_meas.append(bb.add(MeasureX(), q=n_f_gt_ng))
            n_f_gt_ngs.append(n_f_gt_ng)
            # Perform updates based on poly[-1]
            poly[-1], v, u = bb.add(
                GF2Addition(self.dtype.bitsize).controlled(), ctrl=poly[-1], x=v, y=u
            )
            poly, n_f = bb.add(GMinusFPoly(self.dtype, len(poly)), poly=poly, n_f=n_f)
            poly = poly.tolist()
            dialog.append(poly.pop())
            n_g = bb.add(AddK(self.quint, -1), x=n_g)
            u = bb.add(GF2ShiftLeft(self.dtype), f=u)
        # Deallocate poly and u
        poly[0] = bb.add(XGate(), q=poly[0])
        bb.free(poly[0])
        bb.free(u)
        junk = bb.add(self.unpartition_bloq, dialog=dialog, n_f_gt_ngs=n_f_gt_ngs, n_f=n_f, n_g=n_g)
        # TODO: Fix phase errors and cleanup if using MBUC based approach
        # for _ in range(2 * self.n - 1):
        #     is_swap = n_f > n_g and poly[-1]
        #     if is_swap:
        #         n_f, n_g = n_g, n_f
        # for c in n_f_gt_ng_meas:
        #     bb.add(Discard(), c=c)
        return {'junk': junk, 'x_by_y': v}
