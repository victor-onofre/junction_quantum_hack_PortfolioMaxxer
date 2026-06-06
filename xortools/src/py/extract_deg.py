import re
import sys
from collections import defaultdict

def parse_counts(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    mode = None
    degree_counts = {'variable': defaultdict(int), 'constraint': defaultdict(int)}

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if "Variable degree counts" in line:
            mode = 'variable'
            continue
        elif "Constraint degree counts" in line:
            mode = 'constraint'
            continue
        if mode is None:
            continue
        degree, count = map(int, re.split(r'\s+', line))
        degree_counts[mode][degree] += count


    return degree_counts

def to_edge_distribution(degree_count_dict):
    total_edges = sum(degree * count for degree, count in degree_count_dict.items())
    edge_dist = {}
    for degree, count in sorted(degree_count_dict.items()):
        weight = (degree * count) / total_edges
        edge_dist[degree] = weight
    return edge_dist

def write_deg_file(var_dist, check_dist, outfile):
    with open(outfile, 'w') as f:
        f.write("# Edge-perspective degree distribution file\n")
        f.write("# Variable node degrees\n")
        var_degrees = sorted(var_dist)
        var_probs = [var_dist[d] for d in var_degrees]
        f.write(" ".join(map(str, var_degrees)) + "\n")
        f.write(" ".join(f"{p:.8f}" for p in var_probs) + "\n")
        f.write("# Check node degrees\n")
        check_degrees = sorted(check_dist)
        check_probs = [check_dist[d] for d in check_degrees]
        f.write(" ".join(map(str, check_degrees)) + "\n")
        f.write(" ".join(f"{p:.8f}" for p in check_probs) + "\n")

def main(input_file, output_file):
    counts = parse_counts(input_file)
    var_edge_dist = to_edge_distribution(counts['constraint'])
    check_edge_dist = to_edge_distribution(counts['variable'])
    write_deg_file(var_edge_dist, check_edge_dist, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python counts_to_deg.py input_counts.txt output.deg")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
