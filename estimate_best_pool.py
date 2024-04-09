import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy, json

def power(x, power):
    if x==0:
        return 0
    return x**power

from objective import func_effective_data_aggregation, func_effective_utility_aggregation


def estimate_best_buckets(a_values, b_values, c_values, num_samples, pool_size, objective, d):
    d = d
    max_buckets = len(b_values)
    # we will store the error for each bucket length
    error = np.zeros(max_buckets)

    b_values = np.array(b_values)
    a_values = np.array(a_values)
    c_values = np.array(c_values)

    #sort largest based on b_values
    sorted_indices = np.argsort(b_values)
    b_values = b_values[sorted_indices]
    a_values = a_values[sorted_indices]
    c_values = c_values[sorted_indices]

    all_acc = []
    for num_buckets in range(1, max_buckets + 1):
    # for num_buckets in range(2, 3):
        b_list = b_values[:num_buckets]
        a_list = a_values[:num_buckets]
        c_list = c_values[:num_buckets]

        total_data_size = pool_size*num_buckets

        if objective == "effective_data":
            c_list = c_list * num_buckets
            error[num_buckets - 1] = func_effective_data_aggregation(num_samples, [a_list, b_list, c_list, d], total_data_size)
        elif objective == "effective_utility":
            c_list = c_list * num_buckets
            error[num_buckets - 1] = func_effective_utility_aggregation(num_samples, [a_list, b_list, c_list, d], total_data_size)

        all_acc.append(1 - error[num_buckets - 1])

    return error, np.argmin(error) + 1, all_acc
    print("best num buckets", np.argmin(error) + 1)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_size", type=int, default=12_800_000)
    parser.add_argument("--path", type=str, default="results/parameter_values.jsonl")
    parser.add_argument("--key", type=str, default="tmars_imagenet1k")
    parser.add_argument("--objective", type=str, default="effective_utility")
    parser.add_argument("--d", type=float, default=0.1)
    args = parser.parse_args()

    found_key = 0
    with open(args.path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] == args.key:
                found_key = 1
                break
        
        a_values = data["a_values"]
        b_values = data["b_values"]
        c_values = data["c_values"]

    assert found_key == 1, "key not found"

    b_values, a_values, c_values = b_values[:-1], a_values[:-1], c_values[:-1]

    all_accs = {"top10": [], "top20": [], "top30": [], "top40": []}
    for num_samples in [32_000_000, 64_000_000, 128_000_000, 640_000_000]:
        error, num_buckets, all_acc = estimate_best_buckets(a_values, b_values, c_values, num_samples, args.pool_size, args.objective, args.d)
        all_accs["top10"].append(all_acc[0])
        all_accs["top20"].append(all_acc[1])
        all_accs["top30"].append(all_acc[2])
        all_accs["top40"].append(all_acc[3])

    # print comman seperated values
    print(",".join([str(x*100) for x in all_accs["top10"]]))
    print(",".join([str(x*100) for x in all_accs["top20"]]))
    print(",".join([str(x*100) for x in all_accs["top30"]]))
    print(",".join([str(x*100) for x in all_accs["top40"]]))