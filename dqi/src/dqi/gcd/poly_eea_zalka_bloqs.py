from typing import Dict, Union, Set, Mapping

import attrs
import numpy as np
from functools import cached_property

from qualtran import (
    Side,
    Signature,
    Register,
    Bloq,
    QGF,
    QBit,
    QUInt,
    SoquetT,
    BloqBuilder,
    CtrlSpec,
)

from qualtran.symbolics import bit_length
from qualtran.resource_counting import SympySymbolAllocator, BloqCountT, BloqCountDictT
from qualtran.simulation.classical_sim import ClassicalValT, ClassicalValRetT
import sympy
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.arithmetic import AddK, Add
from qualtran.bloqs.basic_gates import Toffoli, XGate
from qualtran.bloqs.gf_arithmetic import GF2Addition
from qualtran.bloqs.data_loading import QROM

from dqi.basic_gates.basic_gates import CSWAP, MCX
from dqi.basic_gates.arithmetic_gates import IsEquals, LessThan, LessThanK, GreaterThanK
from dqi.gf_arithmetic.gf_division_using_flt import GF2DivisionUsingFLT
from dqi.gf_arithmetic.gf_multiplication import GF2MulViaOptimizedPolyMul as GF2MulViaKaratsuba


@attrs.frozen
class _StopBloqBase(Bloq):
    m: int

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('n_A', dtype=self.quint_logm),
                Register('n_B', dtype=self.quint_logm),
                Register('n_a', dtype=self.quint_logm),
                Register('n_b', dtype=self.quint_logm),
                Register('is_stop', dtype=QBit(), side=Side.RIGHT),
            ]
        )


@attrs.frozen
class StopEEA_RS(_StopBloqBase):
    m: int
    n: int

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        n_A, n_B = soqs['n_A'], soqs['n_B']

        # is_stop = (n_A == 0) or (n_B < self.n // 2)
        # is_stop = NOT (NOT (n_A == 0) AND NOT (n_B < self.n // 2))

        # 1. Compute the two conditions into ancillas
        n_A, is_nA_zero = bb.add(IsEquals(self.quint_logm, 0), x=n_A)
        n_B, is_nB_lt_half_n = bb.add(LessThanK(self.quint_logm, self.n // 2), x=n_B)

        # 2. AND the negated conditions into a temporary ancilla
        (is_nA_zero, is_nB_lt_half_n), is_stop = bb.add(
            And(0, 0), ctrl=[is_nA_zero, is_nB_lt_half_n]
        )
        is_stop = bb.add(XGate(), q=is_stop)

        # 3. Uncompute the temporary ancillas by running the gates in reverse
        n_A = bb.add(IsEquals(self.quint_logm, 0).adjoint(), x=n_A, is_equal=is_nA_zero)
        n_B = bb.add(
            LessThanK(self.quint_logm, self.n // 2).adjoint(), x=n_B, target=is_nB_lt_half_n
        )

        soqs.update({'n_A': n_A, 'n_B': n_B, 'is_stop': is_stop})
        return soqs


@attrs.frozen
class StopEEA_LFSR(_StopBloqBase):
    m: int

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        n_A, n_B, n_b = soqs['n_A'], soqs['n_B'], soqs['n_b']

        # is_stop = (n_A == 0) or (n_B <= n_b)
        # is_stop = NOT (NOT (n_A == 0) AND (n_B > n_b))

        # 1. Compute the two conditions into ancillas
        n_A, is_nA_zero = bb.add(IsEquals(self.quint_logm, 0), x=n_A)
        n_b, n_B, is_nb_lt_nB = bb.add(LessThan(self.quint_logm), x=n_b, y=n_B)

        # 2. AND the negated conditions into a temporary ancilla
        (is_nA_zero, is_nb_lt_nB), is_stop = bb.add(And(0, 1), ctrl=[is_nA_zero, is_nb_lt_nB])
        is_stop = bb.add(XGate(), q=is_stop)

        # 3. Uncompute the temporary ancillas by running the gates in reverse
        n_A = bb.add(IsEquals(self.quint_logm, 0).adjoint(), x=n_A, is_equal=is_nA_zero)
        n_b, n_B = bb.add(LessThan(self.quint_logm).adjoint(), x=n_b, y=n_B, target=is_nb_lt_nB)

        soqs.update({'n_A': n_A, 'n_B': n_B, 'n_b': n_b, 'is_stop': is_stop})
        return soqs


@attrs.frozen
class UpdateFlag(Bloq):
    qgf: QGF
    m: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('b_array', dtype=self.qgf, shape=(self.m,)),
                Register('n_A', dtype=self.quint_logm),
                Register('n_B', dtype=self.quint_logm),
                Register('n_a', dtype=self.quint_logm),
                Register('n_b', dtype=self.quint_logm),
                Register('n_q', dtype=QUInt(self.logm)),
                Register('counter', dtype=QUInt(2)),
                Register('flag', dtype=QBit()),
                Register('is_hc_zero', dtype=QBit()),
            ]
        )

    def my_static_costs(self, cost_key: "CostKey"):
        from qualtran.resource_counting import QubitCount, get_cost_value

        if isinstance(cost_key, QubitCount):
            return self.signature.n_qubits() + 4

        return NotImplemented

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        counter, b_array = vals['counter'], vals['b_array']
        n_q, n_A, n_B, n_a, n_b = vals["n_q"], vals["n_A"], vals["n_B"], vals["n_a"], vals["n_b"]
        if counter == 0:
            is_first = (n_q == 0) and (n_A < n_B)
            is_last = n_A == n_B
        elif counter == 1:
            is_first = n_A == n_B
            is_last = n_B == 1 or b_array[-2] != 0
        elif counter == 2:
            is_first = b_array[0] != 0 or n_b == 0
            is_last = n_a == n_b + 1
        elif counter == 3:
            is_first = n_a == n_b
            is_last = n_q == 1
        else:
            assert False
        if vals["is_hc_zero"]:
            vals["flag"] ^= is_first ^ is_last
        return vals

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        # Approx costs since it's a very small contribution.
        return {IsEquals(self.quint_logm, 0): 7, IsEquals(self.qgf, 0): 2, Toffoli(): 10}

    def adjoint(self) -> 'UpdateFlag':
        return self


@attrs.frozen
class _GetSwapIf(Bloq):
    qgf: QGF
    m: int
    i: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('n_q', dtype=QUInt(self.logm)),
                Register('counter', dtype=QBit(), shape=(2,)),
                Register('is_hc_zero', dtype=QBit()),
                Register('swap_if_one', dtype=QBit(), side=Side.RIGHT),
                Register('swap_if_two', dtype=QBit(), side=Side.RIGHT),
                Register('swap_if', dtype=QBit(), side=Side.RIGHT),
            ]
        )

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        counter, is_hc_zero = soqs.pop("counter"), soqs.pop("is_hc_zero")
        n_q = soqs.pop("n_q")

        n_q, is_i_eq_nq = bb.add(IsEquals(self.quint_logm, self.i), x=n_q)
        counter[0] = bb.add(XGate(), q=counter[0])
        counter[1] = bb.add(XGate(), q=counter[1])

        mcx_ctrls = [is_i_eq_nq, counter[0], counter[1], is_hc_zero]
        mcx_ctrls, swap_if_one = bb.add(MCX(4), ctrls=mcx_ctrls)
        [is_i_eq_nq, counter[0], counter[1], is_hc_zero] = mcx_ctrls

        counter[1] = bb.add(XGate(), q=counter[1])
        n_q = bb.add(IsEquals(self.quint_logm, self.i).adjoint(), x=n_q, is_equal=is_i_eq_nq)
        # swap_if_two = (i+1 == n_a + n_q and counter == 0 and halting_counter == 0)
        n_q, is_i_eq_nq = bb.add(IsEquals(self.quint_logm, self.i + 1), x=n_q)

        mcx_ctrls = [is_i_eq_nq, counter[0], counter[1], is_hc_zero]
        mcx_ctrls, swap_if_two = bb.add(MCX(4), ctrls=mcx_ctrls)
        [is_i_eq_nq, counter[0], counter[1], is_hc_zero] = mcx_ctrls

        counter[0] = bb.add(XGate(), q=counter[0])
        n_q = bb.add(IsEquals(self.quint_logm, self.i + 1).adjoint(), x=n_q, is_equal=is_i_eq_nq)
        # swap_if = swap_if_one or swap_if_two = ~(~swap_if_one and ~swap_if_one)
        (swap_if_one, swap_if_two), swap_if = bb.add(And(0, 0), ctrl=[swap_if_one, swap_if_two])
        swap_if = bb.add(XGate(), q=swap_if)
        return {
            "n_q": n_q,
            "counter": counter,
            "is_hc_zero": is_hc_zero,
            "swap_if_one": swap_if_one,
            "swap_if_two": swap_if_two,
            "swap_if": swap_if,
        }


@attrs.frozen
class _GetAddIf(Bloq):
    qgf: QGF
    m: int
    i: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('n_a', dtype=QUInt(self.logm)),
                Register('counter', dtype=QBit(), shape=(2,)),
                Register('add_if', dtype=QBit(), side=Side.RIGHT),
            ]
        )

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        soqs["n_a"], n_a_gt_i = bb.add(GreaterThanK(self.quint_logm, self.i), x=soqs["n_a"])
        ctrls = [*soqs["counter"], n_a_gt_i]
        ctrls, soqs["add_if"] = bb.add(MCX(cvs=(1, 1, 1)), ctrls=ctrls)
        soqs["counter"] = ctrls[:2]
        soqs["n_a"] = bb.add(
            GreaterThanK(self.quint_logm, self.i).adjoint(), x=soqs["n_a"], target=ctrls[-1]
        )
        return soqs


@attrs.frozen
class _GetSubIf(Bloq):
    qgf: QGF
    m: int
    i: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('n_A', dtype=QUInt(self.logm)),
                Register('counter', dtype=QBit(), shape=(2,)),
                Register('sub_if', dtype=QBit(), side=Side.RIGHT),
            ]
        )

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        soqs["n_A"], n_A_lt_m_i = bb.add(LessThanK(self.quint_logm, self.m - self.i), x=soqs["n_A"])
        ctrls = [*soqs["counter"], n_A_lt_m_i]
        ctrls, soqs["sub_if"] = bb.add(MCX(cvs=(0, 0, 0)), ctrls=ctrls)
        soqs["counter"] = ctrls[:2]
        soqs["n_A"] = bb.add(
            LessThanK(self.quint_logm, self.m - self.i).adjoint(), x=soqs["n_A"], target=ctrls[-1]
        )
        return soqs


@attrs.frozen
class MultiplyAndAddOrSub(Bloq):
    qgf: QGF
    m: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('a_array', dtype=self.qgf, shape=(self.m,)),
                Register('b_array', dtype=self.qgf, shape=(self.m,)),
                Register('n_A', dtype=self.quint_logm),
                Register('n_a', dtype=self.quint_logm),
                Register('n_q', dtype=QUInt(self.logm)),
                Register('counter', dtype=QUInt(2)),
                Register('is_hc_zero', dtype=QBit()),
            ]
        )

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    def my_static_costs(self, cost_key: "CostKey"):
        from qualtran.resource_counting import QubitCount, get_cost_value

        if isinstance(cost_key, QubitCount):
            div_ancilla = get_cost_value(GF2DivisionUsingFLT(self.qgf), QubitCount())
            div_ancilla -= 2 * self.qgf.bitsize  # Remove inputs since already counted.
            loop_ancilla = 4 * self.logm + self.qgf.bitsize + 1
            return self.signature.n_qubits() + max(div_ancilla, loop_ancilla) + self.qgf.bitsize

        return NotImplemented

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        a, b = soqs.pop("a_array"), soqs.pop("b_array")
        counter, is_hc_zero = soqs.pop("counter"), soqs.pop("is_hc_zero")
        n_a, n_A, n_q = soqs.pop("n_a"), soqs.pop("n_A"), soqs.pop("n_q")
        assert not soqs
        q_i = bb.allocate(dtype=self.qgf)
        counter = bb.split(counter)

        # Compute b[-1] // a[-1] division
        junk, b_by_a = bb.add(GF2DivisionUsingFLT(self.qgf), x=b[-1], y=a[-1])
        ctrl_add_bloq = GF2Addition(self.qgf.bitsize).controlled(CtrlSpec(cvs=(0, 0, 1)))
        (counter[0], counter[1], is_hc_zero), b_by_a, q_i = bb.add(
            ctrl_add_bloq, ctrl=[counter[0], counter[1], is_hc_zero], x=b_by_a, y=q_i
        )
        b[-1], a[-1] = bb.add(GF2DivisionUsingFLT(self.qgf).adjoint(), junk=junk, x_by_y=b_by_a)

        # Start the loop
        n_a, n_q = bb.add(Add(self.quint_logm), a=n_a, b=n_q)
        for i in range(self.m - 1, -1, -1):
            # Do SWAP
            n_q, counter, is_hc_zero, swap_if_one, swap_if_two, swap_if = bb.add(
                _GetSwapIf(self.qgf, self.m, i), n_q=n_q, counter=counter, is_hc_zero=is_hc_zero
            )
            swap_if, q_i, a[i] = bb.add(CSWAP(self.qgf), ctrl=swap_if, x=q_i, y=a[i])
            n_q, counter, is_hc_zero = bb.add(
                _GetSwapIf(self.qgf, self.m, i).adjoint(),
                n_q=n_q,
                counter=counter,
                is_hc_zero=is_hc_zero,
                swap_if_one=swap_if_one,
                swap_if_two=swap_if_two,
                swap_if=swap_if,
            )
            # Do multiplication
            q_i, a[i], q_i_times_a_i = bb.add(GF2MulViaKaratsuba(self.qgf), x=q_i, y=a[i])
            # Do controlled addition
            n_a, counter, add_if = bb.add(_GetAddIf(self.qgf, self.m, i), n_a=n_a, counter=counter)
            n_A, counter, sub_if = bb.add(_GetSubIf(self.qgf, self.m, i), n_A=n_A, counter=counter)
            (add_if, sub_if), add_or_sub_if = bb.add(And(0, 0), ctrl=[add_if, sub_if])
            add_or_sub_if = bb.add(XGate(), q=add_or_sub_if)
            add_or_sub_if, q_i_times_a_i, b[i] = bb.add(
                GF2Addition(self.qgf.bitsize).controlled(),
                ctrl=add_or_sub_if,
                x=q_i_times_a_i,
                y=b[i],
            )
            add_or_sub_if = bb.add(XGate(), q=add_or_sub_if)
            (add_if, sub_if) = bb.add(
                And(0, 0).adjoint(), ctrl=[add_if, sub_if], target=add_or_sub_if
            )
            n_a, counter = bb.add(
                _GetAddIf(self.qgf, self.m, i).adjoint(), n_a=n_a, counter=counter, add_if=add_if
            )
            n_A, counter = bb.add(
                _GetSubIf(self.qgf, self.m, i).adjoint(), n_A=n_A, counter=counter, sub_if=sub_if
            )
            # Undo multiplication
            q_i, a[i] = bb.add(
                GF2MulViaKaratsuba(self.qgf).adjoint(), x=q_i, y=a[i], result=q_i_times_a_i
            )

        n_a, n_q = bb.add(Add(self.quint_logm).adjoint(), a=n_a, b=n_q)

        # Uncompute q_i
        junk, b_by_a = bb.add(GF2DivisionUsingFLT(self.qgf), x=b[0], y=a[0])
        ctrl_add_bloq = GF2Addition(self.qgf.bitsize).controlled(CtrlSpec(cvs=(1, 1, 1)))
        (counter[0], counter[1], is_hc_zero), b_by_a, q_i = bb.add(
            ctrl_add_bloq, ctrl=[counter[0], counter[1], is_hc_zero], x=b_by_a, y=q_i
        )
        b[0], a[0] = bb.add(GF2DivisionUsingFLT(self.qgf).adjoint(), junk=junk, x_by_y=b_by_a)
        bb.free(q_i)

        return {
            "a_array": a,
            "b_array": b,
            "n_A": n_A,
            "n_a": n_a,
            "n_q": n_q,
            "counter": bb.join(counter),
            "is_hc_zero": is_hc_zero,
        }

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {
            GF2MulViaKaratsuba(self.qgf): self.m,
            GF2MulViaKaratsuba(self.qgf).adjoint(): self.m,
            CSWAP(self.qgf): self.m,
            GF2Addition(self.qgf.bitsize).controlled(): self.m + 2,
            GF2DivisionUsingFLT(self.qgf): 2,
            GF2DivisionUsingFLT(self.qgf).adjoint(): 2,
            # Need to use unary iteration to reduce Toffoli counts for other
            # comparisons.
            QROM.build_from_bitsize((self.m,), (1,), num_controls=1): 6,
        }

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        counter, is_hc_zero = vals['counter'], vals['is_hc_zero']
        a_array, b_array = vals['a_array'], vals['b_array']
        n_a, n_A, n_q = vals['n_a'], vals['n_A'], vals['n_q']
        is_case_zero = counter == 0 and is_hc_zero
        is_case_three = counter == 3 and is_hc_zero

        q_i = 0
        if is_case_zero:
            q_i ^= (a_array[-1] ** -1) * b_array[-1]

        for i in range(self.m - 1, -1, -1):
            swap_if = i == n_a + n_q and is_case_zero
            swap_if |= i + 1 == n_a + n_q and is_case_three
            if swap_if:
                q_i, a_array[i] = a_array[i], q_i

            q_i_times_a_i = q_i * a_array[i]
            add_if = is_case_three and i < n_a
            sub_if = is_case_zero and n_A >= self.m - i
            if add_if or sub_if:
                b_array[i] += q_i_times_a_i

        if is_case_three:
            q_i ^= b_array[0] * (a_array[0] ** -1)
        assert q_i == 0
        return {
            'a_array': a_array,
            'b_array': b_array,
            'n_A': n_A,
            'n_a': n_a,
            'n_q': n_q,
            'counter': counter,
            'is_hc_zero': is_hc_zero,
        }


@attrs.frozen
class ShiftBArrayRight(Bloq):
    qgf: QGF
    m: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('b_array', dtype=self.qgf, shape=(self.m,)),
                Register('n_B', dtype=self.quint_logm),
                Register('n_b', dtype=self.quint_logm),
                Register('flag', dtype=QBit()),
                Register('counter', dtype=QUInt(2)),
                Register('is_hc_zero', dtype=QBit()),
            ]
        )

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {
            CSWAP(self.qgf): self.m,
            QROM.build_from_bitsize((self.m,), (1,), num_controls=1): 3,
            Toffoli(): 4,
        }

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        counter, is_hc_zero, flag = vals['counter'], vals['is_hc_zero'], vals['flag']
        b_array, n_b, n_B = vals['b_array'], vals['n_b'], vals['n_B']

        for i in range(self.m - 1, -1, -1):
            do_shift = False
            if counter == 0 and not flag:
                do_shift = i >= self.m - n_B + 1
            if counter == 1:
                do_shift = i >= self.m - n_B + 1
            if counter == 2:
                do_shift = 0 < i <= n_b
            if counter == 3 and not flag:
                do_shift = 0 < i < self.m - n_B
            if do_shift and is_hc_zero:
                b_array[i - 1], b_array[i] = b_array[i], b_array[i - 1]
        return {
            'b_array': b_array,
            'n_B': n_B,
            'n_b': n_b,
            'flag': flag,
            'counter': counter,
            'is_hc_zero': is_hc_zero,
        }


@attrs.frozen
class UpdateMetadata(Bloq):
    qgf: QGF
    m: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('n_B', dtype=QUInt(self.logm)),
                Register('n_b', dtype=QUInt(self.logm)),
                Register('n_q', dtype=QUInt(self.logm)),
                Register('flag', dtype=QBit()),
                Register('counter', dtype=QUInt(2)),
                Register('is_hc_zero', dtype=QBit()),
            ]
        )

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {AddK(self.quint_logm, 1).controlled(): 6}

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        print(f'{vals=}')
        counter, is_hc_zero, flag = vals['counter'], vals['is_hc_zero'], vals['flag']
        n_b, n_B, n_q = vals['n_b'], vals['n_B'], vals['n_q']

        if counter == 0 and is_hc_zero:
            n_q += 1
            if not flag:
                n_B -= 1
        if counter == 1 and is_hc_zero:
            n_B -= 1
        if counter == 2 and is_hc_zero:
            n_b += 1
        if counter == 3 and is_hc_zero:
            if not flag:
                n_b += 1
            n_q -= 1

        return {
            'n_B': n_B,
            'n_b': n_b,
            'n_q': n_q,
            'flag': flag,
            'counter': counter,
            'is_hc_zero': is_hc_zero,
        }


@attrs.frozen
class SwapAandB(Bloq):
    qgf: QGF
    m: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('a_array', dtype=self.qgf, shape=(self.m,)),
                Register('b_array', dtype=self.qgf, shape=(self.m,)),
                Register('n_A', dtype=self.quint_logm),
                Register('n_B', dtype=self.quint_logm),
                Register('n_a', dtype=self.quint_logm),
                Register('n_b', dtype=self.quint_logm),
                Register('counter', dtype=QUInt(2)),
                Register('flag', dtype=QBit()),
                Register('is_hc_zero', dtype=QBit()),
            ]
        )

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    @cached_property
    def is_equal_counter_3_bloq(self) -> IsEquals:
        return IsEquals(QUInt(2), 3)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        soqs["counter"] = bb.split(soqs["counter"])
        ctrls = [*soqs["counter"], soqs["flag"], soqs["is_hc_zero"]]
        ctrls, swap_ctrl = bb.add(MCX((1, 1, 1, 1)), ctrls=ctrls)

        swap_ctrl, soqs["n_a"], soqs["n_b"] = bb.add(
            CSWAP(self.quint_logm), ctrl=swap_ctrl, x=soqs["n_a"], y=soqs["n_b"]
        )
        swap_ctrl, soqs["n_A"], soqs["n_B"] = bb.add(
            CSWAP(self.quint_logm), ctrl=swap_ctrl, x=soqs["n_A"], y=soqs["n_B"]
        )

        a, b = soqs["a_array"], soqs["b_array"]
        for i in range(self.m):
            swap_ctrl, a[i], b[i] = bb.add(CSWAP(self.qgf), ctrl=swap_ctrl, x=a[i], y=b[i])
        soqs["a_array"], soqs["b_array"] = a, b

        ctrls = bb.add(MCX((1, 1, 1, 1)).adjoint(), ctrls=ctrls, target=swap_ctrl)
        soqs["counter"], soqs["flag"], soqs["is_hc_zero"] = bb.join(ctrls[:2]), ctrls[2], ctrls[3]
        return soqs

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        if vals["counter"] == 3 and vals["flag"] and vals["is_hc_zero"] == 1:
            vals["n_A"], vals["n_B"] = vals["n_B"], vals["n_A"]
            vals["n_a"], vals["n_b"] = vals["n_b"], vals["n_a"]
            vals["a_array"], vals["b_array"] = vals["b_array"], vals["a_array"]
        return vals


@attrs.frozen
class UpdateHaltingCounter(Bloq):
    qgf: QGF
    m: int
    n_steps: int
    stop_bloq: _StopBloqBase

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('n_A', dtype=self.quint_logm),
                Register('n_B', dtype=self.quint_logm),
                Register('n_a', dtype=self.quint_logm),
                Register('n_b', dtype=self.quint_logm),
                Register('flag', dtype=QBit()),
                Register('counter', dtype=QUInt(2)),
                Register('halting_counter', dtype=QUInt(self.log_nsteps)),
            ]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        n_A, n_B, n_a, n_b = soqs['n_A'], soqs['n_B'], soqs['n_a'], soqs['n_b']
        flag, counter, halting_counter = soqs['flag'], soqs['counter'], soqs['halting_counter']

        n_A, n_B, n_a, n_b, is_stop = bb.add(self.stop_bloq, n_A=n_A, n_B=n_B, n_a=n_a, n_b=n_b)

        counter = bb.split(counter)
        mcx_ctrls = [*counter, flag, is_stop]
        mcx_ctrls, update_hc_ctrl = bb.add(MCX((1, 1, 1, 1)), ctrls=mcx_ctrls)

        update_hc_ctrl, halting_counter = bb.add(
            AddK(QUInt(self.log_nsteps), 1).controlled(), ctrl=update_hc_ctrl, x=halting_counter
        )

        mcx_ctrls = bb.add(MCX((1, 1, 1, 1)).adjoint(), ctrls=mcx_ctrls, target=update_hc_ctrl)
        counter[0], counter[1], flag, is_stop = mcx_ctrls
        counter = bb.join(counter, dtype=QUInt(2))

        n_A, n_B, n_a, n_b = bb.add(
            self.stop_bloq.adjoint(), n_A=n_A, n_B=n_B, n_a=n_a, n_b=n_b, is_stop=is_stop
        )

        return {
            'n_A': n_A,
            'n_B': n_B,
            'n_a': n_a,
            'n_b': n_b,
            'flag': flag,
            'counter': counter,
            'halting_counter': halting_counter,
        }

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    @cached_property
    def log_nsteps(self) -> int:
        return int(np.ceil(np.log2(self.n_steps)))

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        is_stop = self.stop_bloq.call_classically(
            n_A=vals['n_A'], n_B=vals['n_B'], n_a=vals['n_a'], n_b=vals['n_b']
        )[-1]
        if vals["counter"] == 3 and vals["flag"] and is_stop:
            vals["halting_counter"] += 1
        return vals


@attrs.frozen
class _PolyEEAZalkaImplStep(Bloq):
    qgf: QGF
    m: int
    n_steps: int
    stop_bloq: _StopBloqBase

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('a_array', dtype=self.qgf, shape=(self.m,)),
                Register('b_array', dtype=self.qgf, shape=(self.m,)),
                Register('n_A', dtype=QUInt(self.logm)),
                Register('n_B', dtype=QUInt(self.logm)),
                Register('n_a', dtype=QUInt(self.logm)),
                Register('n_b', dtype=QUInt(self.logm)),
                Register('n_q', dtype=QUInt(self.logm)),
                Register('flag', dtype=QBit()),
                Register('counter', dtype=QUInt(2)),
                Register('halting_counter', dtype=QUInt(self.log_nsteps)),
            ]
        )

    @cached_property
    def log_nsteps(self) -> int:
        return int(np.ceil(np.log2(self.n_steps)))

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    @cached_property
    def swap_a_and_b(self) -> SwapAandB:
        return SwapAandB(self.qgf, self.m)

    @cached_property
    def update_flag(self) -> UpdateFlag:
        return UpdateFlag(self.qgf, self.m)

    @cached_property
    def multiply_and_add_or_sub(self) -> MultiplyAndAddOrSub:
        return MultiplyAndAddOrSub(self.qgf, self.m)

    @cached_property
    def shift_b_array_right(self) -> ShiftBArrayRight:
        return ShiftBArrayRight(self.qgf, self.m)

    @cached_property
    def update_metadata(self) -> UpdateMetadata:
        return UpdateMetadata(self.qgf, self.m)

    @cached_property
    def update_halting_counter(self) -> UpdateHaltingCounter:
        return UpdateHaltingCounter(self.qgf, self.m, self.n_steps, self.stop_bloq)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {
            self.update_flag: 1,
            self.multiply_and_add_or_sub: 1,
            self.shift_b_array_right: 1,
            self.update_metadata: 1,
            self.swap_a_and_b: 1,
            self.update_halting_counter: 1,
            AddK(QUInt(2), 1).controlled(): 1,
            IsEquals(QUInt(self.log_nsteps), 0): 1,
            IsEquals(QUInt(self.log_nsteps), 0).adjoint(): 1,
        }

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        soqs['halting_counter'], soqs['is_hc_zero'] = bb.add(
            IsEquals(QUInt(self.log_nsteps), 0), x=soqs['halting_counter']
        )

        # 1. Update Flag
        update_flag_soqs = {reg.name: soqs[reg.name] for reg in self.update_flag.signature.lefts()}
        out_soqs = bb.add(self.update_flag, **update_flag_soqs)
        for i, reg in enumerate(self.update_flag.signature.rights()):
            soqs[reg.name] = out_soqs[i]

        # 2. Multiply and Add/Sub
        multiply_soqs = {
            reg.name: soqs[reg.name] for reg in self.multiply_and_add_or_sub.signature.lefts()
        }
        out_soqs = bb.add(self.multiply_and_add_or_sub, **multiply_soqs)
        for i, reg in enumerate(self.multiply_and_add_or_sub.signature.rights()):
            soqs[reg.name] = out_soqs[i]

        # 3. Shift B Array Right
        shift_soqs = {
            reg.name: soqs[reg.name] for reg in self.shift_b_array_right.signature.lefts()
        }
        out_soqs = bb.add(self.shift_b_array_right, **shift_soqs)
        for i, reg in enumerate(self.shift_b_array_right.signature.rights()):
            soqs[reg.name] = out_soqs[i]

        # 4. Update Metadata
        metadata_soqs = {reg.name: soqs[reg.name] for reg in self.update_metadata.signature.lefts()}
        out_soqs = bb.add(self.update_metadata, **metadata_soqs)
        for i, reg in enumerate(self.update_metadata.signature.rights()):
            soqs[reg.name] = out_soqs[i]

        # 5. Swap A and B
        swap_soqs = {reg.name: soqs[reg.name] for reg in self.swap_a_and_b.signature.lefts()}
        out_soqs = bb.add(self.swap_a_and_b, **swap_soqs)
        for i, reg in enumerate(self.swap_a_and_b.signature.rights()):
            soqs[reg.name] = out_soqs[i]

        soqs['halting_counter'] = bb.add(
            IsEquals(QUInt(self.log_nsteps), 0).adjoint(),
            x=soqs['halting_counter'],
            is_equal=soqs.pop('is_hc_zero'),
        )
        # 6. Check Stopping condition
        update_hc_soqs = {
            reg.name: soqs[reg.name] for reg in self.update_halting_counter.signature.lefts()
        }
        out_soqs = bb.add(self.update_halting_counter, **update_hc_soqs)
        for i, reg in enumerate(self.update_halting_counter.signature.rights()):
            soqs[reg.name] = out_soqs[i]

        # 7. counter = (counter + flag) % 4
        soqs['flag'], soqs['counter'] = bb.add(
            AddK(QUInt(2), 1).controlled(), ctrl=soqs['flag'], x=soqs['counter']
        )
        return soqs


@attrs.frozen
class _PolyEEAZalkaImpl(Bloq):
    qgf: QGF
    m: int
    n_steps: int
    stop_bloq: _StopBloqBase

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('a_array', dtype=self.qgf, shape=(self.m,)),
                Register('b_array', dtype=self.qgf, shape=(self.m,)),
                Register('n_A', dtype=QUInt(self.logm)),
                Register('n_B', dtype=QUInt(self.logm)),
                Register('n_a', dtype=QUInt(self.logm)),
                Register('n_b', dtype=QUInt(self.logm)),
                Register('n_q', dtype=QUInt(self.logm)),
                Register('flag', dtype=QBit()),
                Register('counter', dtype=QUInt(2)),
                Register('halting_counter', dtype=QUInt(self.log_nsteps)),
            ]
        )

    @cached_property
    def log_nsteps(self) -> int:
        return int(np.ceil(np.log2(self.n_steps)))

    @cached_property
    def logm(self) -> int:
        return bit_length(self.m)

    @cached_property
    def quint_logm(self) -> QUInt:
        return QUInt(self.logm)

    @cached_property
    def poly_eea_zalka_impl_step(self) -> _PolyEEAZalkaImplStep:
        return _PolyEEAZalkaImplStep(self.qgf, self.m, self.n_steps, self.stop_bloq)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {self.poly_eea_zalka_impl_step: self.n_steps}

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        for _ in range(self.n_steps):
            soqs = bb.add_d(self.poly_eea_zalka_impl_step, **soqs)
        return soqs
