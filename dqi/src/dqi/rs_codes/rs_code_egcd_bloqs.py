from typing import Dict, Union, Set, Mapping

import attrs
from functools import cached_property
from galois import Array

import numpy as np
from numpy.typing import NDArray

from qualtran import Bloq, QGF, Signature, Register, Side, QBit, QInt, bloq_example
from qualtran.symbolics import SymbolicInt, bit_length
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.resource_counting import SympySymbolAllocator, BloqCountT, BloqCountDictT
from qualtran.bloqs.basic_gates import CSwap, XGate
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.arithmetic import AddK, BitwiseNot, LessThanConstant
from qualtran.bloqs.gf_arithmetic import GF2AddK, GF2Addition, GF2MulK
from dqi.basic_gates.arithmetic_gates import IsEquals
from dqi.gf_arithmetic.gf_inverse import GF2InverseOpt as GF2Inverse
from dqi.gf_arithmetic.gf_multiplication import GF2MulViaOptimizedPolyMul as GF2MulViaKaratsuba


@attrs.frozen
class CtrlReversePoly(Bloq):
    """CSWAPs to reverse an array of length `n`."""

    n: SymbolicInt
    dtype: QGF

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('ctrl', dtype=QBit()), Register('poly', dtype=self.dtype, shape=(self.n,))]
        )

    def on_classical_vals(self, ctrl: int, poly: NDArray[object]) -> Dict[str, 'ClassicalValT']:
        return {'ctrl': ctrl, 'poly': poly[::-1] if ctrl else poly}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {CSwap(self.dtype.bitsize): self.n // 2}

    def adjoint(self) -> 'CtrlReversePoly':
        return self


@attrs.frozen
class OmegaEval(Bloq):
    n: SymbolicInt
    dtype: QGF
    k: SymbolicInt = attrs.field(eq=False)
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register('delta', dtype=self.logn_qint),
                Register('poly', dtype=self.dtype, shape=(self.n + 4)),
                Register('omega_eval', dtype=self.dtype, side=side),
            ]
        )

    @cached_property
    def logn_qint(self) -> QInt:
        return QInt(self.logn)

    @cached_property
    def logn(self) -> SymbolicInt:
        return bit_length(self.n)

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __str__(self) -> str:
        """Delegate to subbloq's `__str__` method."""
        cls_name = type(self).__name__
        return f'{cls_name}†' if self.is_adjoint else f'{cls_name}'

    def on_classical_vals(
        self, delta: int, poly: NDArray[object], *, omega_eval=None
    ) -> Dict[str, 'ClassicalValT']:
        omega_eval_x = self.dtype.gf_type(0)
        c = self.dtype.gf_type(self.k)
        cpow = self.dtype.gf_type(1)
        for j in range(self.n // 2):
            if delta // 2 < self.n // 2 - j:
                omega_eval_x += poly[-j - 1] * cpow
            cpow = cpow * c
        if self.is_adjoint:
            assert omega_eval == omega_eval_x
            return {'delta': delta, 'poly': poly}
        else:
            return {'delta': delta, 'poly': poly, 'omega_eval': omega_eval_x}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        gf2_mul_const = GF2MulK(self.dtype, self.k)
        return {
            LessThanConstant(self.logn_qint.bitsize, self.n // 2): self.n // 2,
            gf2_mul_const: 2 * (self.n // 2),
            GF2Addition(self.dtype.bitsize).controlled(): self.n // 2,
        }


@attrs.frozen
class SigmaAndSigmaPrimeEval(Bloq):
    n: SymbolicInt
    dtype: QGF
    k: SymbolicInt = attrs.field(eq=False)
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register('delta', dtype=self.logn_qint),
                Register('is_swaps', dtype=QBit(), shape=(self.n - 1)),
                Register('dialog', dtype=self.dtype, shape=(self.n - 1)),
                Register('eval_sigma', dtype=self.dtype, side=side),
                Register('eval_sigma_prime', dtype=self.dtype, side=side),
                Register('eval_w', dtype=self.dtype, side=side),
                Register('eval_w_prime', dtype=self.dtype, side=side),
            ]
        )

    @cached_property
    def logn_qint(self) -> QInt:
        return QInt(self.logn)

    @cached_property
    def logn(self) -> SymbolicInt:
        return bit_length(self.n)

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __str__(self) -> str:
        """Delegate to subbloq's `__str__` method."""
        cls_name = type(self).__name__
        return f'{cls_name}†' if self.is_adjoint else f'{cls_name}'

    def on_classical_vals(
        self,
        delta: int,
        is_swaps: NDArray[bool],
        dialog: NDArray[object],
        eval_sigma=None,
        eval_sigma_prime=None,
        eval_w=None,
        eval_w_prime=None,
    ) -> Dict[str, 'ClassicalValT']:
        gf = self.dtype.gf_type
        c = gf(self.k)
        eval_w_x, eval_sigma_x = gf(0), gf(1)
        eval_w_x_prime, eval_sigma_x_prime = gf(0), gf(0)
        for coeff, is_swap in zip(dialog, is_swaps):
            if is_swap:  # CSWAP
                eval_w_x, eval_sigma_x = eval_sigma_x, eval_w_x
                eval_w_x_prime, eval_sigma_x_prime = eval_sigma_x_prime, eval_w_x_prime
            eval_w_x_prime_times_coeff = eval_w_x_prime * coeff  # GF2Mul
            eval_sigma_x_prime -= eval_w_x_prime_times_coeff  # GF2Add
            eval_sigma_x_prime = eval_sigma_x_prime * c  # GF2MulK
            eval_sigma_x_prime += eval_sigma_x  # GF2Add
            eval_w_x_prime_times_coeff -= eval_w_x_prime * coeff  # GF2MulAdjoint
            assert eval_w_x_prime_times_coeff == 0

            eval_w_x_times_coeff = eval_w_x * coeff  # GF2Mul
            eval_sigma_x_prime -= eval_w_x_times_coeff  # GF2Add
            eval_sigma_x -= eval_w_x_times_coeff  # GF2Add
            eval_sigma_x = eval_sigma_x * c  # GF2MulK
            eval_w_x_times_coeff -= eval_w_x * coeff  # GF2MulAdjoint
            assert eval_w_x_prime_times_coeff == 0
        ct = self.n // 2 + delta // 2
        eval_sigma_x = eval_sigma_x // c
        eval_sigma_x_prime -= ct * eval_sigma_x
        if self.is_adjoint:
            assert eval_sigma == eval_sigma_x and eval_sigma_prime == eval_sigma_x_prime
            assert eval_w == eval_w_x and eval_w_prime == eval_w_x_prime
            return {'delta': delta, 'is_swaps': is_swaps, 'dialog': dialog}
        else:
            return {
                'delta': delta,
                'is_swaps': is_swaps,
                'dialog': dialog,
                'eval_sigma': eval_sigma_x,
                'eval_sigma_prime': eval_sigma_x_prime,
                'eval_w': eval_w_x,
                'eval_w_prime': eval_w_x_prime,
            }

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        n_iterations = self.n - 1
        return {
            CSwap(self.dtype.bitsize): 2 * n_iterations,
            GF2MulViaKaratsuba(self.dtype.bitsize): 2 * n_iterations + 1,
            GF2MulViaKaratsuba(self.dtype.bitsize).adjoint(): 2 * n_iterations + 1,
            GF2MulK(self.dtype, self.k): 2 * n_iterations + 1,
            GF2Addition(self.dtype.bitsize): 4 * n_iterations + 1,
            AddK(self.logn_qint, self.n // 2): 2,
        }


@attrs.frozen
class GF2Division(Bloq):
    """Computes result = x // y"""

    dtype: QGF
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register('x', self.dtype),
                Register('y', self.dtype),
                Register('result', self.dtype, side=side),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, x: 'SoquetT', y: 'SoquetT', result: 'SoquetT' = None
    ) -> Dict[str, 'SoquetT']:
        if self.is_adjoint:
            y, y_inv, y_inv_junk = bb.add(GF2Inverse(self.dtype.bitsize), x=y)
            x, y_inv = bb.add(
                GF2MulViaKaratsuba(self.dtype.bitsize).adjoint(), x=x, y=y_inv, result=result
            )
            y = bb.add(GF2Inverse(self.dtype.bitsize).adjoint(), x=y, result=y_inv, junk=y_inv_junk)
            return {'x': x, 'y': y}
        else:
            y, y_inv, y_inv_junk = bb.add(GF2Inverse(self.dtype.bitsize), x=y)
            x, y_inv, result = bb.add(GF2MulViaKaratsuba(self.dtype.bitsize), x=x, y=y_inv)
            y = bb.add(GF2Inverse(self.dtype.bitsize).adjoint(), x=y, result=y_inv, junk=y_inv_junk)
        return {'x': x, 'y': y, 'result': result}

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __str__(self) -> str:
        """Delegate to subbloq's `__str__` method."""
        cls_name = type(self).__name__
        return f'{cls_name}†' if self.is_adjoint else f'{cls_name}'

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        result = vals['x'] // vals['y'] if vals['y'] else self.dtype.gf_type(0)
        if self.is_adjoint:
            assert vals.pop('result') == result
        else:
            vals['result'] = result
        return vals


@attrs.frozen
class OmegaMinusU(Bloq):
    """Perform omega_x -= coeff * u_x in shared register representation as part of computing the dialog."""

    poly_len: SymbolicInt
    dtype: QGF
    n: SymbolicInt
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('poly', self.dtype, shape=(self.poly_len,)),
                Register('delta', self.logn_qint),
                Register('coeff', self.dtype)
            ]
        )
    @cached_property
    def logn_qint(self) -> QInt:
        return QInt(self.logn)

    @cached_property
    def logn(self) -> SymbolicInt:
        return bit_length(self.n)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, poly: 'SoquetT', coeff: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        for j in range(len(poly) // 2 - 1, -1, -1):
            poly[j], coeff, result = bb.add(
                GF2MulViaKaratsuba(self.dtype.bitsize), x=poly[j], y=coeff
            )
            bb.add(LessThanConstant(delta))
            result, poly[-j - 1] = bb.add(GF2Addition(self.dtype.bitsize), x=result, y=poly[-j - 1])
            poly[j], coeff = bb.add(
                GF2MulViaKaratsuba(self.dtype.bitsize).adjoint(), x=poly[j], y=coeff, result=result
            )
        return {'poly': poly, 'coeff': coeff}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {
            GF2MulViaKaratsuba(self.dtype.bitsize): self.poly_len // 2,
            GF2MulViaKaratsuba(self.dtype.bitsize).adjoint(): self.poly_len // 2,
            GF2Addition(self.dtype.bitsize).controlled(): self.poly_len // 2,
            Toffoli(): self.poly_len // 2 - 1
        }

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        poly, coeff, delta = vals.pop('poly'), vals.pop('coeff'), vals.pop('delta')
        if self.is_adjoint:
            for j in range(len(poly) // 2):
                if j < len(poly) // 2 + delta // 2:
                    poly[-j - 1] += poly[j] * coeff
        else:
            for j in range(len(poly) // 2 - 1, -1, -1):
                if j < len(poly) // 2 + delta // 2:
                    poly[-j - 1] -= poly[j] * coeff
        return {'poly': poly, 'coeff': coeff, 'delta': delta}

    def adjoint(self):
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)


@attrs.frozen
class ConstructDialog(Bloq):
    n: SymbolicInt
    dtype: QGF
    n_iterations: SymbolicInt = attrs.field()

    @n_iterations.default
    def _default_n_iterations(self):
        return self.n - 1

    @cached_property
    def signature(self) -> 'Signature':
        regs = [
            Register('syndrome', shape=(self.n,), dtype=self.dtype, side=Side.LEFT),
            Register('delta', dtype=self.logn_qint, side=Side.RIGHT),
            Register(
                'poly',
                dtype=self.dtype,
                shape=(2 * self.n + 3 - self.n_iterations,),
                side=Side.RIGHT,
            ),
            Register('dialog', dtype=self.dtype, shape=(self.n_iterations,), side=Side.RIGHT),
            Register('is_swaps', dtype=QBit(), shape=(self.n_iterations,), side=Side.RIGHT),
        ]
        return Signature(regs)

    @cached_property
    def logn_qint(self) -> QInt:
        return QInt(self.logn)

    @cached_property
    def logn(self) -> SymbolicInt:
        return bit_length(self.n)

    def omega_minus_u(self, poly_len: int) -> OmegaMinusU:
        return OmegaMinusU(poly_len, self.dtype, self.n)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, syndrome: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        # 1. Initialize
        poly = [bb.allocate(dtype=self.dtype) for _ in range(self.n + 3)]
        poly = np.concatenate([poly, syndrome])
        poly[0] = bb.add(GF2AddK(self.dtype.bitsize, k=1), x=poly[0])
        delta = bb.allocate(dtype=self.logn_qint)
        dialog, is_swaps = [], []
        # 2. Construct dialog representation.
        for i in range(self.n_iterations):
            poly[-1], is_coeff_zero = bb.add(IsEquals(self.dtype, 0), x=poly[-1])
            delta = bb.split(delta)
            # Checking delta >= 0 is same as checking delta[0] == 0 because of 2s
            # complement representation.
            (is_coeff_zero, delta[0]), is_swap = bb.add(
                And(cv1=0, cv2=0), ctrl=[is_coeff_zero, delta[0]]
            )
            delta = bb.join(delta, dtype=self.logn_qint)
            poly[-1] = bb.add(IsEquals(self.dtype, 0).adjoint(), x=poly[-1], is_equal=is_coeff_zero)
            # delta = -delta - 1 = ~delta
            is_swap, delta = bb.add(BitwiseNot(self.logn_qint).controlled(), ctrl=is_swap, x=delta)
            # Reverse polynomial
            is_swap, poly = bb.add(CtrlReversePoly(len(poly), self.dtype), ctrl=is_swap, poly=poly)
            # Ideally, we can uncompute is_swap via measurement based uncomputation!
            # delta = 1 + delta
            delta = bb.add(AddK(self.logn_qint, k=1), x=delta)
            # Compute coeff using GF2 Inverse. We could use dialog representation here as well!
            poly[-1], poly[0], coeff = bb.add(GF2Division(self.dtype), x=poly[-1], y=poly[0])
            poly, delta, coeff = bb.add(OmegaMinusU(len(poly), self.dtype, self.n), poly=poly, delta=delta, coeff=coeff)
            # poly[-1] should now be 0, so we can free it.
            bb.free(poly[-1])
            poly = poly[:-1]
            dialog.append(coeff)
            is_swaps.append(is_swap)
        # Done!!
        assert len(poly) == 2 * self.n + 3 - self.n_iterations
        return {'delta': delta, 'poly': poly, 'dialog': dialog, 'is_swaps': is_swaps}


@attrs.frozen
class ConstructError(Bloq):
    m: SymbolicInt
    n: SymbolicInt
    dtype: QGF
    index: SymbolicInt
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register('delta', dtype=self.logn_qint, side=Side.THRU),
                Register('poly', dtype=self.dtype, shape=(self.n + 4), side=Side.THRU),
                Register('dialog', dtype=self.dtype, shape=(self.n - 1), side=Side.THRU),
                Register('is_swaps', dtype=QBit(), shape=(self.n - 1), side=Side.THRU),
                Register('is_eval_sigma_zero', dtype=QBit(), side=side),
                Register('error_val', dtype=self.dtype, side=side),
                Register('junk', dtype=self.dtype, shape=(5,), side=side),
            ]
        )

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
    def sigma_eval_bloq(self) -> SigmaAndSigmaPrimeEval:
        return SigmaAndSigmaPrimeEval(self.n, self.dtype, int(self.gammas[self.index]))

    @cached_property
    def omega_eval_bloq(self) -> OmegaEval:
        j = self.m - self.index if self.index else self.index
        return OmegaEval(self.n, self.dtype, int(self.gammas[j]))

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        *,
        delta: 'SoquetT',
        poly: 'SoquetT',
        dialog: 'SoquetT',
        is_swaps: 'SoquetT',
        is_eval_sigma_zero: 'SoquetT' = None,
        error_val: 'SoquetT' = None,
        junk: 'SoquetT' = None,
    ) -> Dict[str, 'SoquetT']:
        i = self.index
        j = self.m - i if i else i
        const_to_mul = self.gammas[j] * (self.gammas[i] ** self.n)
        if self.is_adjoint:
            assert junk is not None
            omega_eval, eval_sigma, eval_sigma_prime, eval_w, eval_w_prime = junk
            error_val = bb.add(GF2MulK(self.dtype, int(const_to_mul**-1)), g=error_val)
            # Hack to ensure that IsEqualZero is processed later by the qubit counting code.
            eval_sigma = bb.join(bb.split(eval_sigma), dtype=self.dtype)
            omega_eval, eval_sigma_prime = bb.add(
                GF2Division(self.dtype).adjoint(),
                x=omega_eval,
                y=eval_sigma_prime,
                result=error_val,
            )
            delta, poly = bb.add(
                self.omega_eval_bloq.adjoint(), delta=delta, poly=poly, omega_eval=omega_eval
            )
            eval_sigma = bb.add(
                IsEquals(self.dtype, 0).adjoint(), x=eval_sigma, is_equal=is_eval_sigma_zero
            )
            delta, is_swaps, dialog = bb.add(
                self.sigma_eval_bloq.adjoint(),
                delta=delta,
                is_swaps=is_swaps,
                dialog=dialog,
                eval_sigma=eval_sigma,
                eval_sigma_prime=eval_sigma_prime,
                eval_w=eval_w,
                eval_w_prime=eval_w_prime,
            )
            soqs = {'delta': delta, 'poly': poly, 'dialog': dialog, 'is_swaps': is_swaps}
        else:
            delta, is_swaps, dialog, eval_sigma, eval_sigma_prime, eval_w, eval_w_prime = bb.add(
                self.sigma_eval_bloq, delta=delta, is_swaps=is_swaps, dialog=dialog
            )
            eval_sigma, is_eval_sigma_zero = bb.add(IsEquals(self.dtype, 0), x=eval_sigma)
            delta, poly, omega_eval = bb.add(self.omega_eval_bloq, delta=delta, poly=poly)
            omega_eval, eval_sigma_prime, error_val = bb.add(
                GF2Division(self.dtype), x=omega_eval, y=eval_sigma_prime
            )
            error_val = bb.add(GF2MulK(self.dtype, int(const_to_mul)), g=error_val)
            soqs = {
                'delta': delta,
                'poly': poly,
                'dialog': dialog,
                'is_swaps': is_swaps,
                'is_eval_sigma_zero': is_eval_sigma_zero,
                'error_val': error_val,
                'junk': np.array([omega_eval, eval_sigma, eval_sigma_prime, eval_w, eval_w_prime]),
            }
        return soqs

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __str__(self) -> str:
        """Delegate to subbloq's `__str__` method."""
        cls_name = type(self).__name__
        return f'{cls_name}†' if self.is_adjoint else f'{cls_name}'


@attrs.frozen
class ConstructOrPhaseErrors(Bloq):
    m: SymbolicInt
    n: SymbolicInt
    dtype: QGF
    include_errors_in_output: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        regs = [
            Register('delta', dtype=self.logn_qint, side=Side.THRU),
            Register('poly', dtype=self.dtype, shape=(self.n + 4), side=Side.THRU),
            Register('dialog', dtype=self.dtype, shape=(self.n - 1), side=Side.THRU),
            Register('is_swaps', dtype=QBit(), shape=(self.n - 1), side=Side.THRU),
        ]
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
    def construct_errors(self) -> list[ConstructError]:
        return [ConstructError(self.m, self.n, self.dtype, i) for i in range(self.m)]

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs) -> Dict[str, 'SoquetT']:
        if self.include_errors_in_output:
            errors = np.array([bb.allocate(dtype=self.dtype) for _ in range(self.m)])
        else:
            errors = []
        for i in range(self.m):
            j = self.m - i if i else i
            soqs = bb.add_d(self.construct_errors[i], **soqs)
            if self.include_errors_in_output:
                soqs['is_eval_sigma_zero'], soqs['error_val'], errors[j] = bb.add(
                    GF2Addition(self.dtype.bitsize).controlled(),
                    ctrl=soqs['is_eval_sigma_zero'],
                    x=soqs['error_val'],
                    y=errors[j],
                )
            soqs = bb.add_d(self.construct_errors[i].adjoint(), **soqs)
        if self.include_errors_in_output:
            soqs |= {"errors": errors}
        return soqs

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        c = ssa.new_symbol('c')
        c = int(self.gammas[1])
        return {self.construct_errors[1]: self.m, self.construct_errors[1].adjoint(): self.m}


@attrs.frozen
class RSCodeEGCD(Bloq):
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
    def construct_dialog_bloq(self) -> ConstructDialog:
        return ConstructDialog(self.n, self.dtype)

    @cached_property
    def construct_or_phase_errors_bloq(self) -> ConstructOrPhaseErrors:
        return ConstructOrPhaseErrors(self.m, self.n, self.dtype, self.include_errors_in_output)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, syndrome: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        # 1. Construct dialog representation
        soqs = bb.add_d(self.construct_dialog_bloq, syndrome=syndrome)
        # 2. Use the dialog representation to compute the errors!
        soqs = bb.add_d(self.construct_or_phase_errors_bloq, **soqs)
        # 3. Uncompute the dialog
        syndrome = bb.add(
            self.construct_dialog_bloq.adjoint(),
            delta=soqs.pop('delta'),
            poly=soqs.pop('poly'),
            dialog=soqs.pop('dialog'),
            is_swaps=soqs.pop('is_swaps'),
        )
        return soqs | {'syndrome': syndrome}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {
            self.construct_dialog_bloq: 1,
            self.construct_or_phase_errors_bloq: 1,
            self.construct_dialog_bloq.adjoint(): 1,
        }


@bloq_example(generalizer=[ignore_split_join, ignore_alloc_free])
def _rs_code_egcd_large_no_errors() -> RSCodeEGCD:
    from qualtran import QGF

    n_constraints = 255
    n_variables = 128
    qgf = QGF(2, 8)

    rs_code_egcd_large_no_errors = RSCodeEGCD(n_constraints, n_variables, qgf)
    return rs_code_egcd_large_no_errors
