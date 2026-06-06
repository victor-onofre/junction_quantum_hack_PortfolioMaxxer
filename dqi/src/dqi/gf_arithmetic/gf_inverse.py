from qualtran import Bloq
from qualtran.bloqs.gf_arithmetic import GF2Inverse
from dqi.gf_arithmetic.gf_multiplication import GF2MulViaOptimizedPolyMul
from functools import cached_property


class GF2InverseOpt(GF2Inverse):
    @cached_property
    def gf2_multiplier(self) -> Bloq:
        return GF2MulViaOptimizedPolyMul(self.qgf)
