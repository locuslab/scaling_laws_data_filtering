import matplotlib.pyplot as plt
import json, sys, os
import re
import numpy as np

from grid_search import grid_search
from plotter import plot_results
import copy


samples_per_step = 4096

keys_of_interest = ["imagenet1k", "cifar10", "vtab/caltech101", "vtab/cifar100", "food101", "imagenet_sketch", "imagenetv2", "imagenet-a", "imagenet-o", "imagenet-r", "objectnet", "vtab/flowers", "vtab/pets", "voc2007", "vtab/resisc45", "cars", "retrieval/flickr_1k_test_image_text_retrieval", "retrieval/mscoco_2014_5k_test_image_text_retrieval"]

def get_avg(json_file_path, keys_of_interest):
    main_metrics = []
    keys_of_interest_local = copy.deepcopy(keys_of_interest)
    with open(json_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data["key"] in keys_of_interest_local:
                main_metrics.append(data["metrics"]["main_metric"])
                keys_of_interest_local.remove(data["key"])
        
    # assert check len of main_metrics is 18
    assert len(main_metrics) == 18
    

    average_main_metric = sum(main_metrics) / len(main_metrics)
    return average_main_metric

def get_accuracy_from_jsonl(args, jsonl_file):
    if args.metric == "18tasks":
        return get_avg(jsonl_file, keys_of_interest)
    
        # assert args.metric == "imagenet1k"
    with open(jsonl_file, 'r') as f:
        main_metric = 0
        for i,line in enumerate(f):
            data = json.loads(line)
            if data.get("key") == args.metric:
                main_metric = data["metrics"]["main_metric"]
    return main_metric

def get_jsonl_files_tmars(folder_name):
    all_paths = []
    for i in [2,4,6,8,10]:
        new_path_to_append = f"eval_results_epoch_{i}_step_-1.jsonl"
        jsonl_path = os.path.join(folder_name, new_path_to_append)
        all_paths.append(jsonl_path)
    return all_paths

def get_jsonl_files_clip(folder_name):
    all_paths = []
    for i in [2,4,6,8,10]:
        new_path_to_append = f"_{i}x/eval_results.jsonl"
        jsonl_path = str(folder_name) + new_path_to_append
        all_paths.append(jsonl_path)
    return all_paths

def get_jsonl_files_random_clip_for_tau_analysis(folder_name):
    all_paths = []
    for i in [1,2,4,8]:
        if "bucket1" in str(folder_name) and i == 1:
            continue
        new_path_to_append = f"_{i}x/eval_results.jsonl"
        jsonl_path = str(folder_name) + new_path_to_append
        if os.path.exists(jsonl_path):
            all_paths.append(jsonl_path)
        else:
            print("Path does not exist: ", jsonl_path)
    return all_paths

def get_jsonl_files_debug(folder_name):
    all_paths = []
    # find all jsonl files in the folder
    for file in os.listdir(folder_name):
        if file.endswith(".jsonl"):
            all_paths.append(os.path.join(folder_name, file))
    return all_paths

def get_jsonl_files(folder_name):
    if "tmars" in str(folder_name):
        return get_jsonl_files_tmars(folder_name)
    elif "random_clip_top50_expts/bucket" in str(folder_name):
        return get_jsonl_files_random_clip_for_tau_analysis(folder_name)
    elif "evals" in str(folder_name):
        return get_jsonl_files_debug(folder_name)
    else:
        return get_jsonl_files_clip(folder_name)

def get_all_results_from_folder(args, data_name, paths, match_with_dict, samples_per_epoch_dict, subsample_every=None):
    folder_path = paths[data_name]
    match_with = match_with_dict[data_name]
    result_dict = {}
    for jsonl_file in get_jsonl_files(folder_path):
        if "tmars" in str(folder_path) or "evals" in str(folder_path):
            match = re.search(r'epoch_(\d+)_', str(jsonl_file))
        else:
            match = re.search(r'_(\d+)x', str(jsonl_file))
        if match:
            epoch_number = int(match.group(1)) 
            step_number = epoch_number * samples_per_epoch_dict[data_name] / samples_per_step           
            result_dict[step_number*samples_per_step] = get_accuracy_from_jsonl(args, jsonl_file)

    result_dict = {k: v for k, v in sorted(result_dict.items())}
    return result_dict
    
def main(args):
    all_results = {}
    # if args.filtering == "tmars":
    from all_paths_128 import paths, alt_name, samples_per_epoch_dict, match_with_dict, subsample_every_dict


    #load objective function
    if args.objective == "effective_data":
        from objective import func_effective_data as func
    elif args.objective == "effective_utility":
        from objective import func_effective_utility as func
    else:
        print("Not implemented")        

    for key in paths.keys():
        res = (get_all_results_from_folder(args, key, paths, match_with_dict, samples_per_epoch_dict, subsample_every_dict[key]))
        all_results[key] = res

    x_vals_dict = {}
    error_vals_dict = {}
    y_vals_dict = {}

    for key in paths.keys():
        x_vals = list(all_results[key].keys())
        y_vals = list(all_results[key].values())
        error_vals = [1 - y_vals[i] for i in range(len(y_vals))]
        x_vals_dict[key] = x_vals
        y_vals_dict[key] = y_vals
        error_vals_dict[key] = error_vals

    def get_params_from_data(data_name, a_upper = None, a = None, tau = None, b  = None, d= None):
        error_vals = error_vals_dict[data_name]
        x_vals = x_vals_dict[data_name]
        samples = samples_per_epoch_dict[data_name]
        samples = [samples for i in range(len(x_vals))]
        loss, popt = grid_search(x_vals, samples, error_vals, func, a_upper = a_upper, a = a, tau = tau, b = b, d = d)
        return loss, popt

    b_values  = []
    a_values = []
    c_values = []


    names = list(alt_name.values())
    fitted_vals_dict = {}
    loss_values = {}

    for i, key in enumerate(paths.keys()):
        if args.verbose:
            print("****** ", key)
        
        print(key, args.a_upper, args.a, args.tau)
        tau_for_bucket = args.tau
        loss, popt = get_params_from_data(key, a_upper = args.a_upper, a = args.a, tau = tau_for_bucket, b = args.b, d=args.d)
        loss_values[key] = loss

        b_values.append(popt[1])
        a_values.append(popt[0])
        c_values.append(popt[2])
        x_vals = x_vals_dict[key]
        samples = samples_per_epoch_dict[key]

        fit = []

        for i in range(len(x_vals)):
            fit.append(func(x_vals[i], popt, samples).item())
        fitted_vals_dict[key] = fit

    print("b_values = ", b_values)
    print("a_values = ", a_values)
    print("c_values = ", c_values)
    print("Avergae Loss = ", np.mean(list(loss_values.values())))
    

    if args.plot:
        plot_results(args, names, paths, x_vals_dict, y_vals_dict, error_vals_dict, fitted_vals_dict, a_values, b_values, c_values, 0.1, samples_per_step)

    return b_values, a_values, c_values, np.mean(list(loss_values.values()))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--a_upper', type=float, default=None)
    parser.add_argument('--metric', type=str, default="imagenet1k")
    parser.add_argument('--a', type=float, default=None)
    parser.add_argument('--plot', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--objective', type=str, default="effective_utility")
    parser.add_argument('--plot_name', type=str, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--b', type=float, default=None)
    parser.add_argument('--d', type=float, default=None)
    parser.add_argument('--k', type=float, default=1)


    args = parser.parse_args()
    main(args)