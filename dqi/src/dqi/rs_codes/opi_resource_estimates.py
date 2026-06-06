import argparse
from dqi.rs_codes.opi_classical_hardness import get_best_runtimes

from qualtran import QGF
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

from dqi.dqi_circuit.dqi_for_opi import DQIforOPIusingEEA


parser = argparse.ArgumentParser('resource estimate for an opi instance')

parser.add_argument(
    '--algo', type=str, required=False, default='zalka', choices=['zalka', 'dialog']
)
parser.add_argument('--subroutine', type=str, required=False, default='dqi', choices=['dqi', 'eea'])
args = parser.parse_args()


PARAMS = [
    (1023, 60, 10, 496),
    (1023, 67, 10, 496),
    (1023, 70, 10, 496),
    (1023, 80, 10, 496),
    (1023, 90, 10, 496),
    (1023, 100, 10, 496),
    (4095, 60, 12, 2016),
    (4095, 67, 12, 2016),
    (4095, 70, 12, 2016),
    (4095, 80, 12, 2016),
    (4095, 90, 12, 2016),
    (4095, 100, 12, 2016),
]


def get_latex_format_string(m, n, b, r, n_toffoli, n_clifford, n_qubits, n_classical_trails):
    return f'$({m}, {n}, {b}, {int(r)})$ & \\num{{{n_toffoli}}} & \\num{{{n_clifford}}} & \\num[exponent-mode=fixed]{{{n_qubits}}} & \\num{{{n_classical_trails}}} &'


def main():
    for m, n, b, r in PARAMS:
        qgf = QGF(2, b)
        classical_runtime = get_best_runtimes(qgf.order, m, [n / m], rs=True, gs=False, r_vals=[r])[
            0
        ]
        r = classical_runtime[0] * qgf.order
        n_classical_trials = classical_runtime[1]
        # Zalka cost
        bloq = DQIforOPIusingEEA(m, n, qgf, use_zalka_eea=args.algo == "zalka")
        if args.subroutine == "eea":
            if args.algo == "zalka":
                bloq = bloq.rs_code_egcd_eea.zalka_eea_bloq
            else:
                bloq = bloq.rs_code_egcd_eea.construct_dialog_bloq
        bloq_counts = get_cost_value(bloq, QECGatesCost())
        n_qubits = get_cost_value(bloq, QubitCount())
        n_toffoli = bloq_counts.total_t_count() // 4
        n_clifford = bloq_counts.clifford
        print(
            get_latex_format_string(m, n, b, r, n_toffoli, n_clifford, n_qubits, n_classical_trials)
        )
        print(f'\\hline')


if __name__ == "__main__":
    main()
