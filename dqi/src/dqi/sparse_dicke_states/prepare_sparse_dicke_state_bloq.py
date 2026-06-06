#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import cached_property
from typing import Dict, Set, Union

import numpy as np
from attrs import frozen
from scipy.special import comb

from qualtran import Bloq, BloqBuilder, Register, Signature, Soquet, SoquetT, QUInt, Side
from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.arithmetic.subtraction import Subtract
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt


@frozen
class SearchForCombinationItem(Bloq):
    r"""Finds the j-th item in a combination corresponding to a given rank.

    This bloq implements the bitwise binary search strategy described in the
    "Iterative Unranking" section of the writeup. It determines the largest
    $c_j$ such that $\binom{c_j}{j} \le r$, where $r$ is the remainder of the
    rank from the previous step.

    This corresponds to the unitary $U_{\text{search}}(m, j)$.

    Args:
        m: The total number of items to choose from.
        j: The index of the combination item to find (from $k$ down to 1).
        m_bitsize: The number of qubits to represent `m`.
        rank_bitsize: The number of qubits to represent the rank.

    Registers:
        rank: The input rank, which is updated to `rank - binom(c_j, j)`.
        c_j: A `m_bitsize` register to store the found combination item $c_j$.
    """

    m: SymbolicInt
    j: SymbolicInt
    m_bitsize: SymbolicInt
    rank_bitsize: SymbolicInt

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('rank', QUInt(self.rank_bitsize)),
                Register('c_j', QUInt(self.m_bitsize), side=Side.RIGHT),
            ]
        )

    @cached_property
    def qrom_bloq(self) -> QROM:
        binom_table = [comb(i, self.j, exact=True) for i in range(self.m + 1)]
        return QROM(
            [np.array(binom_table)],
            selection_bitsizes=(self.m_bitsize,),
            target_bitsizes=(self.rank_bitsize,),
            num_controls=1,
        )

    def my_static_costs(self, cost_key: 'CostKey'):
        from qualtran.resource_counting import QubitCount

        if isinstance(cost_key, QubitCount):
            return 2 * self.m_bitsize + 2 * self.rank_bitsize

        return NotImplemented

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {
            self.qrom_bloq: 2,
            LessThanEqual(self.rank_bitsize, self.rank_bitsize): self.m_bitsize,
            Subtract(QUInt(self.rank_bitsize)).controlled(): self.m_bitsize,
        }


@frozen
class IterativeUnrank(Bloq):
    r"""Unranks a number into its k-combination using an iterative search.

    This bloq implements the $U_\text{Unrank}(m, k)$ unitary. It is composed of
    $k$ sequential calls to `SearchForCombinationItem` to find each of the $k$
    items in the combination, $c_k, c_{k-1}, \dots, c_1$.

    Args:
        m: The total number of items to choose from.
        k: The number of items in the combination.

    Registers:
        rank: A register holding the rank $r$, $0 \le r < \binom{m}{k}$.
        combinations: $k$ registers of size $\log_2 m$ to store the combination.
    """

    m: SymbolicInt
    k: SymbolicInt

    @cached_property
    def m_bitsize(self) -> SymbolicInt:
        return bit_length(self.m)

    @cached_property
    def rank_bitsizes(self) -> list[SymbolicInt]:
        rank_bitsizes = [0] * (self.k + 1)
        for i in range(self.k, 0, -1):
            rank_bitsizes[i] = bit_length(comb(self.m, i, exact=True))
        return rank_bitsizes

    @cached_property
    def num_states(self) -> SymbolicInt:
        return comb(self.m, self.k, exact=True)

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('rank', QUInt(self.rank_bitsizes[-1])),
                Register('combinations', QUInt(self.m_bitsize), shape=(self.k,), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, rank: Soquet) -> Dict[str, SoquetT]:
        combinations = np.empty(self.k, dtype=object)
        all_rank = bb.split(rank)
        for j in range(int(self.k), 0, -1):
            search_bloq = SearchForCombinationItem(
                m=self.m, j=j, m_bitsize=self.m_bitsize, rank_bitsize=self.rank_bitsizes[j]
            )
            curr_rank = bb.join(
                all_rank[: self.rank_bitsizes[j]], dtype=QUInt(self.rank_bitsizes[j])
            )
            curr_rank, combinations[j - 1] = bb.add(search_bloq, rank=curr_rank)
            all_rank[: self.rank_bitsizes[j]] = bb.split(curr_rank)

        return {'rank': bb.join(all_rank), 'combinations': combinations}


@frozen
class PrepareSparseDickeState(Bloq):
    r"""Prepares a sparse Dicke state $\ket{SD^m_k}$.

    This circuit first creates a uniform superposition over ranks
    $r \in \{0, \dots, \binom{m}{k}-1\}$, and then uses the `IterativeUnrank`
    bloq to map this superposition to the desired sparse Dicke state.

    $$
    \ket{0} \rightarrow \frac{1}{\sqrt{\binom{m}{k}}} \sum_{r=0}^{\binom{m}{k}-1} \ket{r}
    \xrightarrow{U_\text{Unrank}}
    \frac{1}{\sqrt{\binom{m}{k}}} \sum_{1 \le c_1 < \dots < c_k \le m} \ket{c_1}\dots\ket{c_k}
    $$

    Args:
        m: The total number of items to choose from.
        k: The number of items in the combination.

    Registers:
        combinations: The $k \times \lceil\log_2 m\rceil$ output register.
    """

    m: SymbolicInt
    k: SymbolicInt

    def __attrs_post_init__(self):
        if not is_symbolic(self.m) and not is_symbolic(self.k):
            if self.k > self.m:
                raise ValueError(f"k must be less than or equal to m, but {self.k} > {self.m}")

    @cached_property
    def m_bitsize(self) -> SymbolicInt:
        return bit_length(self.m)

    @cached_property
    def rank_bitsize(self) -> SymbolicInt:
        return bit_length(self.num_states)

    @cached_property
    def num_states(self) -> SymbolicInt:
        return comb(self.m, self.k, exact=True)

    @property
    def signature(self) -> Signature:
        return Signature(
            [Register('combinations', QUInt(self.m_bitsize), shape=(self.k,), side=Side.RIGHT)]
        )

    def build_composite_bloq(self, bb: BloqBuilder) -> Dict[str, SoquetT]:
        # 1. Prepare uniform superposition over ranks
        rank = bb.allocate(self.rank_bitsize)
        rank = bb.add(PrepareUniformSuperposition(n=self.num_states), target=rank)

        # 2. Apply the unranking unitary
        unrank = IterativeUnrank(m=self.m, k=self.k)
        rank, combinations = bb.add(unrank, rank=rank)

        # 3. Free the rank register
        bb.free(rank)

        return {'combinations': combinations}
