from dataclasses import dataclass, asdict
from typing import Dict, List

try:
    from qiskit_optimization import QuadraticProgram
except ImportError as exc:
    QuadraticProgram = None  # type: ignore
    _QISKIT_IMPORT_ERROR = exc


@dataclass
class MaxLINSATConstraint:
    coefficients: List[int]
    allowed_values_mod_q: List[int]
    modulus: int


@dataclass
class MaxLINSATInstance:
    modulus: int
    num_variables: int
    constraints: List[MaxLINSATConstraint]
    objective_weights: Dict[int, int]


class MaxLINSATTranspiler:
    def __init__(self, modulus: int = 101, scale: int = 10_000):
        """
        modulus: prime q for F_q
        scale: factor to clear decimals (e.g. 10_000 for 4 decimal places)
        """
        if modulus <= 1:
            raise ValueError("Modulus must be an integer greater than 1.")

        self.q = modulus
        self.scale = scale

    def _scale(self, x: float) -> int:
        return int(round(x * self.scale))

    def _allowed_mod_q_values(self, min_value: int, max_value: int) -> List[int]:
        if min_value > max_value:
            return []

        residues = {value % self.q for value in range(min_value, max_value + 1)}
        return sorted(residues)

    def _canonical_sense(self, sense) -> str:
        if hasattr(sense, "name"):
            return sense.name
        return str(sense)

    def _pad_coefficients(self, coefficients: List[int], target_length: int) -> List[int]:
        if len(coefficients) < target_length:
            coefficients = coefficients + [0] * (target_length - len(coefficients))
        return coefficients

    def transpile(self, qp: QuadraticProgram) -> MaxLINSATInstance:
        """
        Transpile a binary QuadraticProgram into a Max-LINSAT instance.
        """
        if QuadraticProgram is None:
            raise ImportError(
                "qiskit_optimization is required to transpile QuadraticProgram objects."
            ) from _QISKIT_IMPORT_ERROR

        if any(v.vartype.name != "BINARY" for v in qp.variables):
            raise ValueError("Only binary variables are supported.")

        num_vars = len(qp.variables)
        constraints: List[MaxLINSATConstraint] = []

        # 1) Transpile linear constraints
        for c in qp.linear_constraints:
            coeffs = [0] * num_vars
            for var, coef in c.linear.to_dict().items():
                idx = qp.variables_index[var]
                coeffs[idx] = self._scale(coef)

            rhs = self._scale(c.rhs)
            sense = self._canonical_sense(c.sense)

            min_sum = sum(b for b in coeffs if b < 0)
            max_sum = sum(b for b in coeffs if b > 0)

            if sense == "EQ":
                allowed = [rhs % self.q]
            elif sense == "LE":
                allowed = self._allowed_mod_q_values(min_sum, rhs)
            elif sense == "GE":
                allowed = self._allowed_mod_q_values(rhs, max_sum)
            else:
                raise ValueError(f"Unsupported constraint sense: {c.sense}")

            constraints.append(
                MaxLINSATConstraint(
                    coefficients=coeffs,
                    allowed_values_mod_q=allowed,
                    modulus=self.q,
                )
            )

        # 2) Linearize quadratic objective terms with auxiliary vars
        obj = qp.objective
        if self._canonical_sense(obj.sense) != "MINIMIZE":
            raise ValueError("Expected a minimization problem.")

        current_num_vars = num_vars
        objective_weights: Dict[int, int] = {}

        for var, coef in obj.linear.to_dict().items():
            idx = qp.variables_index[var]
            objective_weights[idx] = objective_weights.get(idx, 0) + self._scale(coef)

        for (v1, v2), coef in obj.quadratic.to_dict().items():
            i = qp.variables_index[v1]
            j = qp.variables_index[v2]
            z_idx = current_num_vars
            current_num_vars += 1

            w = self._scale(coef)
            objective_weights[z_idx] = objective_weights.get(z_idx, 0) + w

            # z <= x_i  -> z - x_i <= 0
            coeffs = [0] * current_num_vars
            coeffs[z_idx] = 1
            coeffs[i] = -1
            constraints.append(
                MaxLINSATConstraint(
                    coefficients=coeffs,
                    allowed_values_mod_q=self._allowed_mod_q_values(-1, 0),
                    modulus=self.q,
                )
            )

            # z <= x_j  -> z - x_j <= 0
            coeffs = [0] * current_num_vars
            coeffs[z_idx] = 1
            coeffs[j] = -1
            constraints.append(
                MaxLINSATConstraint(
                    coefficients=coeffs,
                    allowed_values_mod_q=self._allowed_mod_q_values(-1, 0),
                    modulus=self.q,
                )
            )

            # x_i + x_j - z <= 1
            coeffs = [0] * current_num_vars
            coeffs[i] = 1
            coeffs[j] = 1
            coeffs[z_idx] = -1
            constraints.append(
                MaxLINSATConstraint(
                    coefficients=coeffs,
                    allowed_values_mod_q=self._allowed_mod_q_values(-1, 1),
                    modulus=self.q,
                )
            )

        # Pad all constraint coefficient vectors to the final number of variables
        for constraint in constraints:
            constraint.coefficients = self._pad_coefficients(
                constraint.coefficients, current_num_vars
            )

        return MaxLINSATInstance(
            modulus=self.q,
            num_variables=current_num_vars,
            constraints=constraints,
            objective_weights=objective_weights,
        )

    def to_dict(self, instance: MaxLINSATInstance) -> Dict:
        return {
            "modulus": instance.modulus,
            "num_variables": instance.num_variables,
            "constraints": [asdict(c) for c in instance.constraints],
            "objective_weights": instance.objective_weights,
        }


if __name__ == "__main__":
    if QuadraticProgram is None:
        raise ImportError(
            "qiskit_optimization is required to execute the example."
        ) from _QISKIT_IMPORT_ERROR

    qp = QuadraticProgram()
    qp.binary_var(name="x0")
    qp.binary_var(name="x1")
    qp.minimize(linear={"x0": 1.0, "x1": 2.0}, quadratic={("x0", "x1"): 3.0})
    qp.linear_constraint(linear={"x0": 1.0, "x1": 1.0}, sense="LE", rhs=1.0)

    transpiler = MaxLINSATTranspiler(modulus=101, scale=10_000)
    instance = transpiler.transpile(qp)
    result = transpiler.to_dict(instance)

    import json

    print(json.dumps(result, indent=2))
