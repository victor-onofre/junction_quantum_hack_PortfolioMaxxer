def load_instance_from_tsv(tsv_fname):
    instance = {}
    with open(tsv_fname, "r") as infile:
        lines = [line.strip() for line in infile if not line.startswith("#")]

    # Parsing the first line for n, m, and p
    n, m = map(int, lines[0].split())
    instance["n"], instance["m"], instance["p"] = n, m, 2

    # Initialize the constraints
    instance["bits"] = []

    # Processing the constraints
    for line in lines[1:]:
        # Splitting the line at ',' to separate RHS value and coefficients
        parts = line.split()
        sum_values = []
        if parts[0][0] == "-":
            sum_values.append(0)
        else:
            sum_values.append(1)
        coeffs = []
        variables = []
        for x in parts[1:]:
            index, coeff = int(x), 1
            assert coeff != 0, "all coeffs should be nonzero"
            coeffs.append(coeff)
            variables.append(index)

        bit_checks = {
            # For binary instances, these coefficients should always be 1
            "coefficients": coeffs,
            # This is the list of parity checks containing this bit
            "checks": variables,
        }

        # Append the constraint to the list
        instance["bits"].append(bit_checks)

    # Verify the number of constraints
    assert len(instance["bits"]) == instance["m"]

    return instance


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python tsv_to_parity_checks.py instance_file_name.tsv")
        exit(1)

    instance = load_instance_from_tsv(sys.argv[1])
    print(f"Instance {instance}")
