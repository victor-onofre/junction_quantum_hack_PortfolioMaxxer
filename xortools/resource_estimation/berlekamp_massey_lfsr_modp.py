import attrs
from functools import cached_property
from typing import Union, Set, Dict
import numpy as np

from qualtran import (
    Bloq,
    Signature,
    Side,
    Register,
    QMontgomeryUInt,
    QBit,
    QUInt,
    CtrlSpec,
    SoquetT,
    BloqBuilder,
    bloq_example,
)
from qualtran.symbolics import SymbolicInt, bit_length
from qualtran.bloqs.arithmetic import LessThanConstant, Xor, BitwiseNot, AddK, XorK
from qualtran.bloqs.mod_arithmetic import (
    ModAddK,
    ModAdd,
    CModAdd,
    DirtyOutOfPlaceMontgomeryModMul,
    CModSub,
    KaliskiModInverse,
)
from qualtran.bloqs.mcmt import MultiControlX, And
from qualtran.resource_counting import BloqCountDictT
from qualtran.symbolics import HasLength
from qualtran.resource_counting.generalizers import ignore_split_join, ignore_alloc_free


@attrs.frozen
class IsEqualsZero(Bloq):
    dtype: QMontgomeryUInt
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> "Signature":
        is_zero_side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register(name="x", dtype=self.dtype),
                Register(name="is_zero", dtype=QBit(), side=is_zero_side),
            ]
        )

    def on_classical_vals(
        self, **vals: Union["sympy.Symbol", "ClassicalValT"]
    ) -> Dict[str, "ClassicalValT"]:
        val_to_set = vals["x"] == 0
        if self.is_adjoint:
            assert np.all(vals.pop("is_zero") == val_to_set)
        else:
            vals["is_zero"] = val_to_set
        return vals

    def adjoint(self) -> "Bloq":
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def build_call_graph(
        self, ssa: "SympySymbolAllocator"
    ) -> Union["BloqCountDictT", Set["BloqCountT"]]:
        return {MultiControlX(HasLength(self.dtype.num_qubits)): 1}


@attrs.frozen
class InPlaceModMul(Bloq):
    r"""|x>|x^{-1}>|y> \rightarrow |x> |x^{-1}> |x * y>

    1. |x> |x^{-1}> |y>                -> |x> |x^{-1}> |y> |x * y>
    2. |x> |x^{-1}> |y> |x * y>        -> |x> |x^{-1}> |y> |x * y> |(x * y) * x^{-1}>
    3. |x> |x^{-1}> |y> |x * y> |y>    -> |x> |x^{-1}> |0> |x * y> |y>
    4. |x> |x^{-1}> |0> |x * y> |y>    -> |x> |x^{-1}> |x * y>|x * y> |y>
    5. |x> |x^{-1}> |0> |x * y> |y>    -> |x> |x^{-1}> |x * y>|x * y> |y>
    6. |x> |x^{-1}> |x * y>|x * y> |y> -> |x> |x^{-1}> |x * y> |y>
    7. |x> |x^{-1}> |x * y> |y>        -> |x> |x^{-1}> |x * y>
    """

    dtype: QMontgomeryUInt

    @cached_property
    def signature(self) -> "Signature":
        return Signature(
            [
                Register(name="x", dtype=self.dtype),
                Register(name="x_inv", dtype=self.dtype),
                Register(name="y", dtype=self.dtype),
            ]
        )

    @cached_property
    def p(self) -> int:
        return self.dtype.modulus

    def build_composite_bloq(
        self, bb: "BloqBuilder", *, x: "SoquetT", x_inv: "SoquetT", y: "SoquetT"
    ) -> Dict[str, "SoquetT"]:
        modp_multiply = DirtyOutOfPlaceMontgomeryModMul(
            self.dtype.bitsize, 1, mod=self.p
        )
        multiply_one_soqs = bb.add_d(modp_multiply, x=x, y=y)
        x, y, xy = (
            multiply_one_soqs.pop("x"),
            multiply_one_soqs.pop("y"),
            multiply_one_soqs.pop("target"),
        )
        multiply_two_soqs = bb.add_d(modp_multiply, x=x_inv, y=xy)
        x_inv, xy, xyx_inv = (
            multiply_two_soqs.pop("x"),
            multiply_two_soqs.pop("y"),
            multiply_two_soqs.pop("target"),
        )
        xyx_inv, y = bb.add(Xor(self.dtype), x=xyx_inv, y=y)
        # |y> is 0 now
        xy, y = bb.add(Xor(self.dtype), x=xy, y=y)
        # |y> stores the desired |x * y> now.
        # First uncompute |x * y>
        x, xyx_inv = bb.add(
            modp_multiply.adjoint(), x=x, y=xyx_inv, target=xy, **multiply_one_soqs
        )
        # Then uncompute xyx_inv
        x_inv, y = bb.add(
            modp_multiply.adjoint(), x=x_inv, y=y, target=xyx_inv, **multiply_two_soqs
        )
        return {"x": x, "x_inv": x_inv, "y": y}


@attrs.frozen
class PlusEqualProdMod(Bloq):
    r"""|x>|y>|z> \rightarrow |x> |y> |(z + x * y) % mod>

    1. |x>|y>|z>                              -> |x>|y>|x * y>|junk>|z>
    2. |x>|y>|x * y>|junk>|z>                 -> |x>|y>|x * y>|junk>|(z + x * y)%mod>
    3. |x>|y>|x * y>|junk>|(z + x * y)%mod>   -> |x>|y>|(z + x * y)%mod>
    """

    dtype: QMontgomeryUInt

    @cached_property
    def signature(self) -> "Signature":
        return Signature(
            [
                Register(name="x", dtype=self.dtype),
                Register(name="y", dtype=self.dtype),
                Register(name="z", dtype=self.dtype),
            ]
        )

    @cached_property
    def p(self) -> int:
        return self.dtype.modulus

    def build_composite_bloq(
        self, bb: "BloqBuilder", *, x: "SoquetT", y: "SoquetT", z: "SoquetT"
    ) -> Dict[str, "SoquetT"]:
        modp_multiply = DirtyOutOfPlaceMontgomeryModMul(
            self.dtype.bitsize, 1, mod=self.p
        )
        mul_soqs = bb.add_d(modp_multiply, x=x, y=y)
        mul_soqs["target"], z = bb.add(
            ModAdd(self.dtype.bitsize, self.p), x=mul_soqs["target"], y=z
        )
        x, y = bb.add(modp_multiply.adjoint(), **mul_soqs)
        return {"x": x, "y": y, "z": z}


@attrs.frozen
class CalcDotProduct(Bloq):
    """|x_0〉|x_1〉 ... |x_n〉|y_0〉|y_1〉 ... |y_n〉--> |x_0〉|x_1〉 ... |x_n〉|y_0〉|y_1〉 ... |y_n〉|s〉

    s = Σ_i={0}^{n} x_{n-i} * y_i
    """

    c_len_max: SymbolicInt
    dtype: QMontgomeryUInt

    @cached_property
    def signature(self) -> "Signature":
        return Signature(
            [
                Register(name="s", dtype=self.dtype, shape=(self.c_len_max,)),
                Register(name="c", dtype=self.dtype, shape=(self.c_len_max,)),
                Register(name="result", dtype=self.dtype, side=Side.RIGHT),
            ]
        )

    @cached_property
    def p(self) -> SymbolicInt:
        return self.dtype.modulus

    def build_composite_bloq(
        self, bb: "BloqBuilder", *, s: "SoquetT", c: "SoquetT"
    ) -> Dict[str, "SoquetT"]:
        result = bb.allocate(dtype=self.dtype)
        for i in range(self.c_len_max):
            j = self.c_len_max - i - 1
            c[i], s[j], result = bb.add(
                PlusEqualProdMod(self.dtype), x=c[i], y=s[j], z=result
            )
        return {"s": s, "c": c, "result": result}


@attrs.frozen
class BerlekampMasseyLFSR(Bloq):
    r"""Bloq to find shortest linear feedback shift register (LFSR) using Berlekamp Massey Algorithm"""

    dtype: QMontgomeryUInt
    num_elements: SymbolicInt
    c_len_max: SymbolicInt = attrs.field()

    @classmethod
    def build(cls, p: int, n: SymbolicInt, c_len_max: SymbolicInt):
        dtype = QMontgomeryUInt(1 + p.bit_length(), p)
        return cls(dtype, n, c_len_max)

    @c_len_max.default
    def _clen_max_default(self):
        return self.num_elements

    @cached_property
    def signature(self) -> "Signature":
        return Signature(
            [
                Register(name="s", dtype=self.dtype, shape=(self.num_elements,)),
                Register(
                    name="c", dtype=self.dtype, shape=(self.c_len_max,), side=Side.RIGHT
                ),
                Register(
                    name="old_c",
                    dtype=self.dtype,
                    shape=(self.c_len_max,),
                    side=Side.RIGHT,
                ),
                Register(
                    name="c_len", dtype=QUInt(self.c_len_bitsize), side=Side.RIGHT
                ),
                Register(
                    name="discrep",
                    dtype=self.dtype,
                    shape=(self.num_elements,),
                    side=Side.RIGHT,
                ),
                Register(
                    name="is_c_len_less_than_i",
                    dtype=QBit(),
                    shape=(self.num_elements,),
                    side=Side.RIGHT,
                ),
            ]
        )

    @cached_property
    def c_len_bitsize(self) -> SymbolicInt:
        return bit_length(self.c_len_max)

    @cached_property
    def p(self) -> int:
        return self.dtype.modulus

    def build_composite_bloq(self, bb: "BloqBuilder", *, s) -> Dict[str, "SoquetT"]:
        c_len = bb.allocate(dtype=QUInt(1))
        c = np.array([bb.allocate(dtype=self.dtype)])
        old_c = np.array([bb.allocate(dtype=self.dtype)])
        c[0] = bb.add(
            ModAddK(self.dtype.bitsize, self.p, self.dtype.uint_to_montgomery(-1)),
            x=c[0],
        )
        old_c[0] = bb.add(
            ModAddK(self.dtype.bitsize, self.p, self.dtype.uint_to_montgomery(-1)),
            x=old_c[0],
        )
        discrep = np.empty((self.num_elements,), dtype=object)
        is_c_len_less_than_i = np.empty((self.num_elements,), dtype=object)
        ratio = bb.allocate(dtype=self.dtype)
        inv_ratio = bb.allocate(dtype=self.dtype)
        for i in range(self.num_elements):
            # Compute discrepency
            calc_dot_prod = CalcDotProduct(len(c), self.dtype)
            s[i + 1 - len(c) : i + 1], c, discrep[i] = bb.add(
                calc_dot_prod, s=s[i + 1 - len(c) : i + 1], c=c
            )
            if i + 1 < self.c_len_max:
                c = np.concatenate([c, [bb.allocate(dtype=self.dtype)]])
                old_c = np.concatenate([[bb.allocate(dtype=self.dtype)], old_c])
                if len(c).bit_length() > c_len.reg.dtype.num_qubits:
                    c_len = bb.join(
                        np.concatenate([[bb.allocate(dtype=QBit())], bb.split(c_len)]),
                        dtype=QUInt(len(c).bit_length()),
                    )
            else:
                old_c = np.roll(old_c, 1)
            c_len_dtype = QUInt(bit_length(len(c)))
            # Add bloqs for i'th iteration of berlekamp massey.
            # Compute is_discrep_zero and is_c_len_less_than_i
            discrep[i], is_discrep_zero = bb.add(IsEqualsZero(self.dtype), x=discrep[i])
            is_c_len_less_than_i[i] = bb.allocate(dtype=QBit())
            less_than_const_bloq = LessThanConstant(c_len_dtype.bitsize, i // 2 + 1)
            c_len, is_c_len_less_than_i[i] = bb.add(
                less_than_const_bloq, x=c_len, target=is_c_len_less_than_i[i]
            )
            # Compute ratio to multiply into the old_c.
            # if is_discrep_zero: ratio += 1
            is_discrep_zero, ratio = bb.add(
                XorK(self.dtype, self.dtype.uint_to_montgomery(1)).controlled(
                    CtrlSpec(cvs=1)
                ),
                ctrl=is_discrep_zero,
                x=ratio,
            )
            # if not is_discrep_zero: ratio += discrep
            is_discrep_zero, discrep[i], ratio = bb.add(
                Xor(self.dtype).controlled(CtrlSpec(cvs=0)),
                ctrl=is_discrep_zero,
                x=discrep[i],
                y=ratio,
            )
            # Case-2: if not is_discrep_zero and is_c_len_less_than_i
            (is_discrep_zero, is_c_len_less_than_i[i]), ctrl_bit = bb.add(
                And(0, 1), ctrl=[is_discrep_zero, is_c_len_less_than_i[i]]
            )
            # c_len = i + 1 - c_len
            ctrl_bit, c_len = bb.add(
                AddK(dtype=c_len_dtype, k=~(i + 1)).controlled(), ctrl=ctrl_bit, x=c_len
            )
            ctrl_bit, c_len = bb.add(
                BitwiseNot(c_len_dtype).controlled(), ctrl=ctrl_bit, x=c_len
            )
            # old_c = add_arrays(c, old_c)
            #  In-place multiplication of ratio into each element of old_c[i]
            #  old_c = multiply_by_constant(ratio, old_c)
            # Compute inverse_ratio
            ratio, inv_ratio = bb.add(Xor(self.dtype), x=ratio, y=inv_ratio)
            inv_ratio_soqs = bb.add_d(
                KaliskiModInverse(self.dtype.bitsize, self.p), x=inv_ratio
            )
            inv_ratio = inv_ratio_soqs.pop("x")
            for j in range(len(old_c)):
                # 1. old_c[j] = old_c[j] * ratio
                ratio, inv_ratio, old_c[j] = bb.add(
                    InPlaceModMul(self.dtype), x=ratio, x_inv=inv_ratio, y=old_c[j]
                )
                # Case-1: if not is_discrep_zero
                # if not is_discrep_zero:
                #      c[j] -= old[j]
                # In-place controlled-subtraction of arrays c and old_c
                is_discrep_zero, old_c[j], c[j] = bb.add(
                    CModSub(dtype=self.dtype, mod=self.p, cv=0),
                    ctrl=is_discrep_zero,
                    x=old_c[j],
                    y=c[j],
                )
                # Case-2: if not is_discrep_zero and is_c_len_less_than_i
                # if not is_discrep_zero and is_c_len_less_than_i:
                #       old_c[j] += c[j]
                ctrl_bit, c[j], old_c[j] = bb.add(
                    CModAdd(self.dtype, mod=self.p, cv=1),
                    ctrl=ctrl_bit,
                    x=c[j],
                    y=old_c[j],
                )
                # Uncontrolled old_c = multiply_by_constant(inverse_ratio, old_c)
                # old_c[j] = old_c[j] * inverse_ratio
                inv_ratio, ratio, old_c[j] = bb.add(
                    InPlaceModMul(self.dtype), x=inv_ratio, x_inv=ratio, y=old_c[j]
                )

            # Uncompute inverse_ratio.
            inv_ratio = bb.add(
                KaliskiModInverse(self.dtype.bitsize, self.p).adjoint(),
                x=inv_ratio,
                **inv_ratio_soqs,
            )
            ratio, inv_ratio = bb.add(Xor(self.dtype), x=ratio, y=inv_ratio)
            # Uncompute ctrl_bit for case-2
            is_discrep_zero, is_c_len_less_than_i[i] = bb.add(
                And(0, 1).adjoint(),
                ctrl=[is_discrep_zero, is_c_len_less_than_i[i]],
                target=ctrl_bit,
            )
            # if is_discrep_zero: ratio += 1
            is_discrep_zero, ratio = bb.add(
                XorK(self.dtype, self.dtype.uint_to_montgomery(1)).controlled(
                    CtrlSpec(cvs=1)
                ),
                ctrl=is_discrep_zero,
                x=ratio,
            )
            # if not is_discrep_zero: ratio += discrep
            is_discrep_zero, discrep[i], ratio = bb.add(
                Xor(self.dtype).controlled(CtrlSpec(cvs=0)),
                ctrl=is_discrep_zero,
                x=discrep[i],
                y=ratio,
            )
            # Uncompute is_discrep_zero
            discrep[i] = bb.add(
                IsEqualsZero(self.dtype).adjoint(),
                x=discrep[i],
                is_zero=is_discrep_zero,
            )
        bb.free(ratio)
        bb.free(inv_ratio)
        return {
            "s": s,
            "c": c,
            "old_c": old_c,
            "c_len": c_len,
            "discrep": discrep,
            "is_c_len_less_than_i": is_c_len_less_than_i,
        }


@bloq_example(generalizer=[ignore_split_join, ignore_alloc_free])
def _berlekamp_massey_lfsr_small() -> BerlekampMasseyLFSR:
    from qualtran import QMontgomeryUInt
    from qualtran.symbolics import bit_length

    N, P = 12, 13
    dtype = QMontgomeryUInt(1 + bit_length(P), P)
    berlekamp_massey_lfsr_small = BerlekampMasseyLFSR(dtype, N, N // 2)
    return berlekamp_massey_lfsr_small


@bloq_example(generalizer=[ignore_split_join, ignore_alloc_free])
def _berlekamp_massey_lfsr_large() -> BerlekampMasseyLFSR:
    from qualtran import QMontgomeryUInt
    from qualtran.symbolics import bit_length

    N, P = 128, 257
    dtype = QMontgomeryUInt(1 + bit_length(P), P)
    berlekamp_massey_lfsr_large = BerlekampMasseyLFSR(dtype, N, N // 2)
    return berlekamp_massey_lfsr_large
