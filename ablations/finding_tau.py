'''
This file calls main() function of process_128_grid.py. multiple times with different values of args.a
Then finds the mean loss of all the buckets
Then reports the a with the minimum mean loss over all the a values
'''
import argparse

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from process_128_grid import main
    # run main(args) with different values of args.a. verbose 0, plot 0, metric imagenet1k, filtering tmars, a_upper None
    # a_values = 10 different values between 0.001 and 0.1
    metric = "imagenet1k"
    filtering = "tmars"
    objective = "effective_utility"
    
    if filtering == "clip":
        tau_values_global = [1,2,3,4,4.5,4.7,5,5.5,5.7,6,6.3,6.5,6.7,7,7.3,8,9,10,11,12,13,14,15,16,17,18,19,20]
    else:
        tau_values_global = [1,2,3,4,5,5.5,5.7,6,6.3,6.5,6.7,7,7.3,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
    
    # retrieved by running finding_a.py
    a_dict = {
        "tmars": {
            "18tasks": 0.033,
            "imagenet1k": 0.022
        },
        "clip": {
            "18tasks": 0.035,
            "imagenet1k": 0.021
        }
    }

    #based on results from finding_a.py
    a = a_dict[filtering][metric]

    loss_values = []
    b_values = []
    c_values = []
    a_values = []
    
    for tau in tau_values_global:
        args = argparse.Namespace(a_upper=None, metric=metric, a=a, filtering=filtering, plot=0, verbose=0, objective=objective, tau = tau)
        b_value, a_value, c_value, loss = main(args)
        loss_values.append(loss)
        b_values.append(b_value)
        c_values.append(c_value)
        a_values.append(a_value)
    print("loss_values = ", loss_values)
    print("a_values = ", tau_values_global)
    
    index_min_loss = loss_values.index(min(loss_values))
    print("best tau = ", tau_values_global[index_min_loss])
    print("best loss = ", min(loss_values))
    print("b_values = ", b_values[index_min_loss])
    print("c_values = ", c_values[index_min_loss])
    print("a_values = ", a_values[index_min_loss])
    print (args)