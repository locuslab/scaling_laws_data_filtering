
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

    
    a_values_global = [0.01, 0.015, 0.018, 0.02, 0.022, 0.025, 0.03]
    
    d_vals = list(np.linspace(0.02, 0.15, 10))

    loss_values = []
    b_values = []
    c_values = []
    a_values = []
    d_values = []
    for d in d_vals:
        for a in a_values_global:
            print("Metric = ", args.metric, "a = ", a,  "d = ", d, "filtering = ", args.filtering, "objective = ", args.objective)
            args = argparse.Namespace(a_upper=None, metric=args.metric, a=a, filtering=args.filtering, plot=0, verbose=0, objective=args.objective, tau = None, b = None, d=d)
            b_value, a_value, c_value, loss = main(args)
            loss_values.append(loss)
            b_values.append(b_value)
            c_values.append(c_value)
            a_values.append(a_value)
            d_values.append(d)
            print("loss_values = ", loss_values)
            print("a_values = ", a_values_global)
            print("d_values = ", d_values)
    
    index_min_loss = loss_values.index(min(loss_values))
    print("best a = ", a_values[index_min_loss][0])
    print("best loss = ", min(loss_values))
    print("b_values = ", b_values[index_min_loss])
    print("c_values = ", c_values[index_min_loss])
    print("a_values = ", a_values[index_min_loss])
    print("d_value = ", d_values[index_min_loss])
    print (args)