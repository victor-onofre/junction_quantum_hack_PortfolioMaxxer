from typing import Dict, Union, Optional, Set

import attrs
import sympy
from functools import cached_property

from qualtran import Side, Signature, Register, Bloq, QBit, QUInt, QDType

from qualtran.resource_counting import SympySymbolAllocator, BloqCountT, BloqCountDictT
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.bloqs.basic_gates import Toffoli

from dqi.basic_gates.basic_gates import MCX


@attrs.frozen
class IsEquals(Bloq):
    dtype: QDType
    k: int
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register(name='x', dtype=self.dtype),
                Register(name='is_equal', dtype=QBit(), side=side),
            ]
        )

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Dict[str, 'ClassicalValT']:
        val_to_set = vals['x'] == self.k
        if self.is_adjoint:
            assert vals.pop('is_equal') == val_to_set
        else:
            vals['is_equal'] = val_to_set
        return vals

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {MCX(int(self.dtype.num_qubits)): 1}

    def __str__(self) -> str:
        return 'IsEquals†' if self.is_adjoint else 'IsEquals'


@attrs.frozen
class LessThan(Bloq):
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

    def on_classical_vals(
        self, x: int, y: int, target: Optional[int] = None
    ) -> Dict[str, 'ClassicalValT']:
        if self.is_adjoint:
            assert target == (x < y)
        return {'x': x, 'y': y} if self.is_adjoint else {'x': x, 'y': y, 'target': x < y}

    def __str__(self) -> str:
        return 'LessThan†' if self.is_adjoint else 'LessThan'


@attrs.frozen
class LessThanK(Bloq):
    r"""Low ancilla bloq for quantum-classical comparison"""

    dtype: QUInt
    k: int
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [Register('x', dtype=self.dtype), Register('target', dtype=QBit(), side=side)]
        )

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def on_classical_vals(self, x: int, target: Optional[int] = None) -> Dict[str, 'ClassicalValT']:
        if self.is_adjoint:
            assert target == (x < self.k)
        return {'x': x} if self.is_adjoint else {'x': x, 'target': x < self.k}

    def __str__(self) -> str:
        return 'LessThanK†' if self.is_adjoint else 'LessThanK'

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {Toffoli(): 2 * self.dtype.bitsize}

    def my_static_costs(self, cost_key: "CostKey"):
        from qualtran.resource_counting import QubitCount

        if isinstance(cost_key, QubitCount):
            return self.signature.n_qubits() + 4
        return NotImplemented


@attrs.frozen
class GreaterThanK(Bloq):
    r"""Low ancilla bloq for quantum-classical comparison"""

    dtype: QUInt
    k: int
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [Register('x', dtype=self.dtype), Register('target', dtype=QBit(), side=side)]
        )

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def on_classical_vals(self, x: int, target: Optional[int] = None) -> Dict[str, 'ClassicalValT']:
        if self.is_adjoint:
            assert target == (x > self.k)
        return {'x': x} if self.is_adjoint else {'x': x, 'target': x > self.k}

    def __str__(self) -> str:
        return 'GreaterThanK†' if self.is_adjoint else 'GreaterThanK'

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {Toffoli(): 2 * self.dtype.bitsize}

    def my_static_costs(self, cost_key: "CostKey"):
        from qualtran.resource_counting import QubitCount

        if isinstance(cost_key, QubitCount):
            return self.signature.n_qubits() + 4
        return NotImplemented
