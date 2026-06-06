from typing import Union, Set

import attrs
from functools import cached_property
from galois import Array

from qualtran import Bloq, QGF, Signature, Register, Side, QInt, QUInt
from qualtran.symbolics import SymbolicInt, bit_length
from qualtran.resource_counting import SympySymbolAllocator, BloqCountT, BloqCountDictT
from qualtran.bloqs.gf_arithmetic import GF2Addition, GF2MulK

from dqi.gcd.poly_eea_zalka_bloqs import _PolyEEAZalkaImpl, IsEquals, StopEEA_RS
from dqi.gf_arithmetic.gf_inverse import GF2InverseOpt as GF2Inverse


@attrs.frozen
class ConstructError(Bloq):
    n: SymbolicInt
    dtype: QGF
    include_errors_in_output: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('b_array', dtype=self.dtype, shape=(self.n + 3), side=Side.THRU),
                Register('n_b', dtype=self.logn_quint),
                Register('n_B', dtype=self.logn_quint),
                Register('error', dtype=self.dtype, side=Side.RIGHT),
            ]
        )

    @cached_property
    def logn_quint(self) -> QUInt:
        return QUInt(self.logn)

    @cached_property
    def logn(self) -> SymbolicInt:
        return bit_length(self.n + 3)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        gamma = self.dtype.gf_type.primitive_element
        n = self.n + 2
        return {
            GF2MulK(self.dtype, int(gamma)): n + 1,
            GF2Addition(self.dtype.bitsize): 3 * n,
            GF2Inverse(self.dtype.bitsize): 1,
            # GF2Multiplication(self.dtype): 1,
            IsEquals(self.dtype, 0): 1,
        }


@attrs.frozen
class ConstructOrPhaseErrors(Bloq):
    m: SymbolicInt
    n: SymbolicInt
    dtype: QGF
    include_errors_in_output: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        regs = [
            Register('b_array', dtype=self.dtype, shape=(self.n + 3), side=Side.THRU),
            Register('n_b', dtype=self.logn_quint),
            Register('n_B', dtype=self.logn_quint),
        ]
        if self.include_errors_in_output:
            regs.append(Register('errors', shape=(self.m,), dtype=self.dtype, side=Side.RIGHT))
        return Signature(regs)

    @cached_property
    def logn_quint(self) -> QUInt:
        return QUInt(self.logn)

    @cached_property
    def logn(self) -> SymbolicInt:
        return bit_length(self.n + 3)

    @cached_property
    def gammas(self) -> Array:
        gamma = self.dtype.gf_type.primitive_element
        return self.dtype.gf_type([gamma**i for i in range(0, self.m)])

    @cached_property
    def construct_error(self) -> ConstructError:
        return ConstructError(self.n, self.dtype)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {self.construct_error: self.m, self.construct_error.adjoint(): self.m}


@attrs.frozen
class RSCodeEGCDZalkaEEA(Bloq):
    """Decoding Reed Solomon codes using EEA.

    Given m constraints and n variables as mxn matrix B, we wish to implement a reversible
    decoder for the dual code with parity check matrix B^{T}. The dual is a reed solomon
    code with parameters RS(m, m - n, n + 1) such that length of the syndrome is n.

    Given |y>|s=B^{T}y>, we wish to uncompute the error register |y> using the syndrome register
    |s>. The size of the input is (m + n) * d and size of the output is n * d. Our strategy will
    be to use measurement based uncomputation to first free up the m * d qubits occupied by the
    error register and then compute the error values one by one via a reversible implementation
    of EEA based syndrome decoder for Reed Solomon codes; and then for each error value perform
    a phase correction depending on measurement result of the error register.

    The total number of qubits needed for reversible syndrome decoder compilation will be
    2 * n * d + O(logN); which is less than the size of the input since n ~= m // 2. :)

    Args:
        m: Number of constraints in DQI.
        n: Number of variables in DQI.
        dtype: Data type of each element in the syndrome.
    """

    m: SymbolicInt
    n: SymbolicInt
    dtype: QGF
    include_errors_in_output: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        regs = [Register('syndrome', shape=(self.n,), dtype=self.dtype, side=Side.THRU)]
        if self.include_errors_in_output:
            regs.append(Register('errors', shape=(self.m,), dtype=self.dtype, side=Side.RIGHT))
        return Signature(regs)

    @cached_property
    def logn_qint(self) -> QInt:
        return QInt(self.logn)

    @cached_property
    def logn(self) -> SymbolicInt:
        return bit_length(self.n)

    @cached_property
    def gammas(self) -> Array:
        gamma = self.dtype.gf_type.primitive_element
        return self.dtype.gf_type([gamma**i for i in range(0, self.m)])

    @cached_property
    def n_eea_steps(self) -> SymbolicInt:
        return 3 * (self.n + 1) + 5 + 4

    @cached_property
    def zalka_eea_bloq(self) -> _PolyEEAZalkaImpl:
        stop_rs = StopEEA_RS(self.n + 3, self.n)
        return _PolyEEAZalkaImpl(
            self.dtype, self.n + 3, n_steps=self.n_eea_steps, stop_bloq=stop_rs
        )

    @cached_property
    def construct_or_phase_erros(self) -> ConstructOrPhaseErrors:
        return ConstructOrPhaseErrors(self.m, self.n, self.dtype)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {
            self.zalka_eea_bloq: 1,
            self.construct_or_phase_erros: 1,
            self.zalka_eea_bloq.adjoint(): 1,
        }
