from typing import Union, Set, Mapping

import attrs
import sympy
import numpy as np
from functools import cached_property

from qualtran import Side, Signature, Register, Bloq, QBit, QDType

from qualtran.resource_counting import SympySymbolAllocator, BloqCountT, BloqCountDictT
from qualtran.simulation.classical_sim import ClassicalValT, ClassicalValRetT
from qualtran.bloqs.basic_gates import Toffoli, TwoBitCSwap


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
class MCX(Bloq):
    cvs: tuple[int, ...] = attrs.field(converter=lambda x: (1,) * x if isinstance(x, int) else x)
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [Register("ctrls", QBit(), shape=(self.n,)), Register("target", QBit(), side=side)]
        )

    @cached_property
    def n(self) -> int:
        return len(self.cvs)

    def my_static_costs(self, cost_key: "CostKey"):
        from qualtran.resource_counting import QubitCount, get_cost_value

        if isinstance(cost_key, QubitCount):
            if self.n > 2:
                return self.signature.n_qubits() + 1
            else:
                return self.signature.n_qubits()

        return NotImplemented

    def short_name(self) -> str:
        name = r"$C^{n}X$"
        return f'{name}†' if self.is_adjoint else f'{name}'

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        val_to_set = np.all(vals['ctrls'] == self.cvs)
        if self.is_adjoint:
            assert vals.pop('target') == val_to_set
        else:
            vals['target'] = val_to_set
        return vals

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {Toffoli(): self.n - 1}

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)
