from glob import glob
import csv
import numpy as np
import os


def load_bndata(fname):
    with open(fname, "r") as infile:
        reader = csv.reader(infile, quotechar="|")
        keys = next(reader)
        data = {k: [] for k in keys}
        for row in reader:
            for k, v in zip(keys, map(int, row)):
                data[k].append(v)

    # convert to numpy
    data = {k: np.array(v) for k, v in data.items()}
    assert len({len(v) for v in data.values()}) == 1
    # sort by p
    idx = np.argsort(data["p"])
    data = {k: v[idx] for k, v in data.items()}
    # sort by num_distractors / p
    idx = np.argsort(data["num_distractors"] / data["p"])
    data = {k: v[idx] for k, v in data.items()}
    return data


def aggregate(globstr, fixed_keys, aggregate_keys):
    aggregated = {}
    for fname in glob(globstr):
        try:
            data = load_bndata(fname)
        except:
            import traceback

            traceback.print_exc()
            continue
        for i in range(data[fixed_keys[0]].shape[0]):
            key = tuple(data[k][i] for k in fixed_keys)
            if key not in aggregated:
                aggregated[key] = {k: 0 for k in aggregate_keys}
            for k in aggregate_keys:
                aggregated[key][k] += data[k][i]
    data = {k: [] for k in fixed_keys + aggregate_keys}
    for key, val in aggregated.items():
        for k, kv in zip(fixed_keys, key):
            data[k].append(kv)
        for k in aggregate_keys:
            data[k].append(val[k])

    # convert to numpy
    data = {k: np.array(v) for k, v in data.items()}
    # sort by p
    idx = np.argsort(data["p"])
    data = {k: v[idx] for k, v in data.items()}
    # sort by num_distractors / p
    idx = np.argsort(data["num_distractors"] / data["p"])
    data = {k: v[idx] for k, v in data.items()}
    return data


def aggregate_to(fname, globstr, fixed_keys, aggregate_keys):
    data = aggregate(globstr, fixed_keys, aggregate_keys)
    with open(fname, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile,
            quotechar="|",
        )
        writer.writerow(data.keys())
        for i in range(data[list(data.keys())[0]].shape[0]):
            writer.writerow([data[k][i] for k in data])


if __name__ == "__main__":
    # fixed_keys = ["p", "m", "n", "num_distractors"]
    # aggregate_keys = ["num_shots", "num_success"]
    # aggregate_to(
    #     "bleichenbacher_nguyen_data/lll.csv",
    #     "bleichenbacher_nguyen_data/data/lll.*.csv",
    #     fixed_keys, aggregate_keys,
    # )
    # aggregate_to(
    #     "bleichenbacher_nguyen_data/bkz.csv",
    #     "bleichenbacher_nguyen_data/data/bkz.*.csv",
    #     fixed_keys, aggregate_keys,
    # )

    os.makedirs('bleichenbacher_nguyen_data', exist_ok=True)
    fixed_keys_num_shots_data = [
        "p",
        "apply_same_sum",
        "m",
        "n",
        "num_distractors",
        "max_num_sat",
        "min_l1",
        "min_l22",
    ]
    aggregate_keys_num_shots_data = ["num_shots"]
    aggregate_to(
        "bleichenbacher_nguyen_data/lll_hist.csv",
        "bleichenbacher_nguyen_data/num_shots_data/*.lll.csv",
        fixed_keys_num_shots_data,
        aggregate_keys_num_shots_data,
    )
    aggregate_to(
        "bleichenbacher_nguyen_data/bkz_hist.csv",
        "bleichenbacher_nguyen_data/num_shots_data/*.bkz.csv",
        fixed_keys_num_shots_data,
        aggregate_keys_num_shots_data,
    )
    aggregate_to(
        "bleichenbacher_nguyen_data/bkz-bs15_hist.csv",
        "bleichenbacher_nguyen_data/num_shots_data/*.bkz-bs15.csv",
        fixed_keys_num_shots_data,
        aggregate_keys_num_shots_data,
    )
