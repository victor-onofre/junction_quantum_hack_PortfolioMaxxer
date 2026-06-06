from typing import Dict, Tuple

import attrs
from functools import cached_property
from qualtran.symbolics import SymbolicInt
from qualtran import Bloq, QGF, Signature, Register, Side, QAny
from qualtran.bloqs.bookkeeping import Partition
from dqi.gf_arithmetic.gf_inverse import GF2InverseOpt as GF2Inverse
from dqi.gf_arithmetic.gf_multiplication import GF2MulViaOptimizedPolyMul as GF2MulViaKaratsuba


@attrs.frozen
class GF2DivisionUsingFLT(Bloq):
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
    def gf2_inverse(self):
        return GF2Inverse(self.dtype.bitsize)

    @cached_property
    def junk_bitsize(self) -> SymbolicInt:
        return sum(reg.total_bits() for reg in self.junk_regs)

    @cached_property
    def junk_regs(self) -> Tuple[Register, ...]:
        return (
            Register('x', self.dtype, side=Side.RIGHT),
            Register('y', self.dtype, side=Side.RIGHT),
            Register('y_inv', self.dtype, side=Side.RIGHT),
            Register(
                'y_inv_junk', self.dtype, shape=(self.gf2_inverse.n_junk_regs,), side=Side.RIGHT
            ),
        )

    @cached_property
    def unpartition_bloq(self) -> Partition:
        return Partition(self.junk_bitsize, self.junk_regs, partition=False)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, x: 'SoquetT', y: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        y, y_inv, y_inv_junk = bb.add(self.gf2_inverse, x=y)
        x, y_inv, x_by_y = bb.add(GF2MulViaKaratsuba(self.dtype), x=x, y=y_inv)
        junk = bb.add(self.unpartition_bloq, x=x, y=y, y_inv=y_inv, y_inv_junk=y_inv_junk)
        return {'junk': junk, 'x_by_y': x_by_y}
