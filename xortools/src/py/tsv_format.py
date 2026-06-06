def load_instance_from_tsv(tsv_fname):
    instance = {}
    with open(tsv_fname, "r") as infile:
        line = next(infile).strip()
        instance["n"], instance["m"] = line.split()
        instance["n"] = int(instance["n"])
        instance["m"] = int(instance["m"])
        instance["constraints"] = []
        for line in infile:
            weight, *b = line.strip().split()
            b = list(map(int, b))
            if weight == "-0.5000":
                targets = [1]
            else:
                assert weight == "0.5000"
                targets = [0]

            instance["constraints"].append(
                {
                    "coefficients": [1] * len(b),
                    "variables": b,
                    "sum_values": targets,
                }
            )
    return instance
