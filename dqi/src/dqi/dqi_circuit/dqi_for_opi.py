import attrs

from qualtran import QGF
from qualtran.resource_counting import SympySymbolAllocator, BloqCountT, BloqCountDictT

from dqi.rs_codes.rs_code_egcd_bloqs_zalka_eea import RSCodeEGCDZalkaEEA
from dqi.rs_codes.rs_code_egcd_bloqs import RSCodeEGCD
from dqi.sparse_dicke_states.prepare_sparse_dicke_state_bloq import PrepareSparseDickeState
from functools import cached_property
from typing import Set, Union

from qualtran import Bloq, Register, Signature, Side
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.symbolics import SymbolicInt


@attrs.frozen
class DQIforOPIusingEEA(Bloq):
    m: SymbolicInt
    n: SymbolicInt
    dtype: QGF
    use_zalka_eea: bool = True

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('syndrome', shape=(self.n,), dtype=self.dtype, side=Side.THRU)])

    @cached_property
    def prepare_sparse_dicke_state(self) -> PrepareSparseDickeState:
        return PrepareSparseDickeState(self.m, self.n // 2)

    @cached_property
    def qrom_for_checking_in_sds(self) -> QROM:
        return QROM.build_from_bitsize((self.m,), (1,))

    @cached_property
    def rs_code_egcd_eea(self) -> RSCodeEGCDZalkaEEA | RSCodeEGCD:
        return (
            RSCodeEGCDZalkaEEA(self.m, self.n, self.dtype)
            if self.use_zalka_eea
            else RSCodeEGCD(self.m, self.n, self.dtype)
        )

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {
            self.prepare_sparse_dicke_state: 1,  # Step-1
            self.qrom_for_checking_in_sds: self.n // 2,  # Step-2
            self.rs_code_egcd_eea: 1,  # Step-3
        }
