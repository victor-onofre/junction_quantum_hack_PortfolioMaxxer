import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate
from typing import Optional
from qualtran import QGF, Controlled, Adjoint, Bloq
from qualtran.symbolics import Shaped
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount
from dqi.dqi_circuit.dqi_for_opi import DQIforOPIusingEEA
from qualtran.bloqs.data_loading import QROM
from dqi.gcd.poly_eea_zalka_bloqs import CSWAP, MCX
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.arithmetic import AddK
from qualtran.bloqs.gf_arithmetic import GF2MulViaKaratsuba, GF2Addition
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit, GF2MulMBUC
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join

parser = argparse.ArgumentParser('resource estimate for an opi instance (m, n, b)')

parser.add_argument('--b', type=int, required=True)
parser.add_argument('--m', type=int, required=True)
parser.add_argument('--n', type=int, required=True)
parser.add_argument(
    '--algo', type=str, required=False, default='zalka', choices=["zalka", "dialog"]
)
args = parser.parse_args()


def keep(bloq: 'Bloq'):
    if isinstance(
        bloq,
        (GF2MulViaKaratsuba, GF2Addition, SynthesizeLRCircuit, CSWAP, AddK, MCX, GF2MulMBUC, QROM),
    ):
        return bloq
    if isinstance(bloq, (Controlled, Adjoint)):
        if keep(bloq.subbloq):
            return bloq
    return None


def stv_rounds(b: int, bloq: 'Bloq'):
    if isinstance(bloq, SynthesizeLRCircuit):
        return 10
    if isinstance(bloq, GF2MulViaKaratsuba):
        return int(np.ceil(1.5 * get_cost_value(bloq, QECGatesCost()).total_toffoli_only()))
    if isinstance(bloq, CSWAP):
        return 6
    if isinstance(bloq, GF2Addition):
        return 2
    if isinstance(bloq, Toffoli):
        return 4
    if isinstance(bloq, GF2MulMBUC):
        return 12
    if isinstance(bloq, MCX):
        return len(bloq.cvs)
    if isinstance(bloq, Controlled) and isinstance(bloq.subbloq, GF2Addition):
        return 8
    if isinstance(bloq, Adjoint):
        return stv_rounds(b, bloq.subbloq)
    if isinstance(bloq, Controlled) and isinstance(bloq.subbloq, Adjoint):
        return stv_rounds(b, Adjoint(Controlled(bloq.subbloq.subbloq, ctrl_spec=bloq.ctrl_spec)))
    return 0


def get_canonicalize_bloqs(n: int):
    def canonicalize_bloqs(b: Bloq) -> Optional[Bloq]:
        """A generalizer that canonicalizes bloqs."""

        if isinstance(b, SynthesizeLRCircuit):
            return SynthesizeLRCircuit(Shaped((b.n, b.n)))

        if isinstance(b, MCX):
            return MCX((1,) * n.bit_length())

        if isinstance(b, AddK):
            return AddK(type(b.dtype)(n.bit_length()), 1)

        if isinstance(b, Controlled):
            return Controlled(canonicalize_bloqs(b.subbloq), b.ctrl_spec)

        if isinstance(b, Adjoint):
            return Adjoint(canonicalize_bloqs(b.subbloq))

        if isinstance(b, QROM):
            return QROM.build_from_bitsize(b.data_shape, (1,))

        return b

    return canonicalize_bloqs


def main():
    bloq = DQIforOPIusingEEA(args.m, args.n, QGF(2, args.b), use_zalka_eea=args.algo == 'zalka')
    g, sigma = bloq.call_graph(
        keep=keep,
        generalizer=[ignore_alloc_free, ignore_split_join, get_canonicalize_bloqs(args.n)],
    )
    df = pd.DataFrame(
        [
            {
                "Bloq": k,
                "Counts": v,
                "STV Rounds": stv_rounds(args.b, k),
                "Toffoli": get_cost_value(k, QECGatesCost()).total_t_count() // 4,
            }
            for k, v in sigma.items()
        ]
    )
    df["Total Toffoli"] = df["Toffoli"] * df["Counts"]
    df["Total Toffoli (%)"] = 100 * df["Total Toffoli"] / df["Total Toffoli"].sum()
    df["Total STV Rounds"] = df["STV Rounds"] * df["Counts"]
    df["Total STV Rounds (%)"] = 100 * df["Total STV Rounds"] / df["Total STV Rounds"].sum()
    print(
        tabulate(
            df.sort_values(by=["Total STV Rounds", "Total Toffoli"]),
            headers='keys',
            tablefmt='psql',
        )
    )
    print(tabulate(df.sort_values(by=["Total Toffoli"]), headers='keys', tablefmt='psql'))
    print(tabulate(df.sort_values(by=["Counts"]), headers='keys', tablefmt='psql'))

    n_toffoli = get_cost_value(bloq, QECGatesCost()).total_t_count() // 4
    n_qubits = get_cost_value(bloq, QubitCount())
    n_stv_rounds = df["Total STV Rounds"].sum()
    print(
        f'------------\nLogical Counts: '
        f'{n_qubits=}, '
        f'{n_toffoli=:.1e}, '
        f'n_toffoli_df={df["Total Toffoli"].sum():.1e}, '
        f'{n_stv_rounds=:.1e}\n------------'
    )


if __name__ == '__main__':
    main()
