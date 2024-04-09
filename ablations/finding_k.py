
import numpy as np
'''
This file calls main() function of process_128_grid.py. multiple times with different values of args.a
Then finds the mean loss of all the buckets
Then reports the a with the minimum mean loss over all the a values
'''
import argparse
keys_of_interest = ["imagenet1k", "cifar10", "vtab/caltech101", "vtab/cifar100", "food101", "imagenet_sketch", "imagenetv2", "imagenet-a", "imagenet-o", "imagenet-r", "objectnet", "vtab/flowers", "vtab/pets", "voc2007", "vtab/resisc45", "cars", "retrieval/flickr_1k_test_image_text_retrieval", "retrieval/mscoco_2014_5k_test_image_text_retrieval"]
if __name__ == "__main__":

    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from process_128_grid import main
    # run main(args) with different values of args.a. verbose 0, plot 0, metric imagenet1k, filtering tmars, a_upper None
    # a_values = 10 different values between 0.001 and 0.1
    # metric = "caltech"
    # filtering = "tmars"
    # # objective = "effective_utility_b_delta"
    # objective = "effective_data"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default="imagenet1k")
    parser.add_argument('--filtering', type=str, default="tmars")
    parser.add_argument('--objective', type=str, default="effective_data")
    args = parser.parse_args()

    
    if args.metric == "imagenet1k":
        # k_values_global = [0.001, 0.01, 0.015, 0.018, 0.019, 0.02, 0.021, 0.022, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1]
        k_values_global = list(np.linspace(0.1, 3, 20))
        # k_values_global = #[5, 7, 10, 11, 12, 15, 20, 25]#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        #make a grid of a with 20 values between 1 and 2
        # k_values_global = np.linspace(0.03, 0.05, 10)
    else:
        k_values_global = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        # k_values_global = np.linspace(0.005, 0.02, 10)
        #convert to list
        k_values_global = list(k_values_global)
        # k_values_global = [0.008, 0.009, 0.01, 0.011, 0.012, 0.015]
    

    loss_values = []
    b_values = []
    c_values = []
    a_values = []
    d_values = []
    for k in k_values_global:
        args = argparse.Namespace(a = 0.022, metric=args.metric, filtering=args.filtering, plot=0, verbose=0, objective=args.objective, tau = 3, b = -0.13106060606060607, d=0.1, k=k, a_upper=None)
        b_value, a_value, c_value, loss = main(args)
        loss_values.append(loss)
    print("Loss values: ", loss_values)
    print("K values: ", k_values_global)