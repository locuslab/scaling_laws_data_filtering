import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def grid_search(x_vals, samples_per_epoch_vals, error_vals, func, a_upper=0.2, a= None, tau = None, b = None, d= None):
    if a is None:
        a = np.linspace(0.0001, a_upper, 100)
    else:
        a = a
    
    if b is None:
        b_lim = np.linspace(-0.005, -0.5, 100)
    else:
        b_lim = b
    
    if tau is None:
        c_lim = np.linspace(1, 10, 100)
    else:
        c_lim = tau

    d = d

    # create a grid of all possible combinations
    grid = np.array(np.meshgrid(a, b_lim, c_lim, d)).T.reshape(-1, 4)
    #randomize the grid
    np.random.shuffle(grid)
    
    # get the best params by running func on all combinations
    # also store the loss grid to plot later of shape b_lim, c_lim
    # output_grid = np.zeros((len(b_lim), len(c_lim)))
    
    best_params = None
    best_loss = 10000
    pbar = tqdm(total=len(grid))
    
    for params in grid:
        loss = 0
        func_values_list = []

        for i in range(len(x_vals)):
            samples_per_epoch = samples_per_epoch_vals[i]
            samples_seen = x_vals[i]
            func_value = func(samples_seen, params, samples_per_epoch)
            func_values_list.append(func_value)
            true_value = error_vals[i]
            curr_loss = (func_value - true_value)**2
            loss += curr_loss
        
        param_1_idx = np.where(b_lim == params[1])[0][0]
        param_2_idx = np.where(c_lim == params[2])[0][0]

        if loss < best_loss:
            best_loss = loss
            best_params = params

        pbar.update(1)
        pbar.set_description("best loss: {} : best params : {}".format(best_loss, best_params))
        if pbar.n == 100_000:
            break

    return best_loss, best_params


def grid_search_from_dict(x_vals_dict, samples_per_epoch_vals_dict, error_vals_dict, func):
    '''
    x_vals: dict of list of lists of x values
    samples_per_epoch_vals: dict of list of samples per epoch
    error_vals: dict of list of error values
    '''
    a = 1
    b_lim_dict = {}
    for key in x_vals_dict.keys():
        b_lim = np.linspace(-0.01, -0.2, 100)
        b_lim_dict[key] = b_lim

    c_lim = np.linspace(1, 100, 100)
    d = np.linspace(0.0, 0.4, 10)

    keys = list(x_vals_dict.keys())

    grid = np.array(np.meshgrid(a, b_lim_dict[keys[0]],b_lim_dict[keys[1]], c_lim, d)).T.reshape(-1, 5)
    grid = np.array(np.meshgrid(a, b_lim_dict[keys[0]],1, c_lim, d)).T.reshape(-1, 5)

    np.random.shuffle(grid)

    output_grid = np.zeros((len(b_lim), len(c_lim)))
    best_params = None
    best_loss = 10000
    pbar = tqdm(total=len(grid))

    for params in grid:
        loss = 0
        for key in keys:
            x_vals, samples_per_epoch_vals, error_vals = x_vals_dict[key], samples_per_epoch_vals_dict[key], error_vals_dict[key]
            b_index = keys.index(key)

            params_current = [params[0], params[1], params[3], params[4]]

            for i in range(len(x_vals)):
                samples_per_epoch = samples_per_epoch_vals[i]
                samples_seen = x_vals[i]
                func_value = func(samples_seen, params_current, samples_per_epoch)
                true_value = error_vals[i]
                curr_loss = (func_value - true_value)**2
                loss += curr_loss
        param_1_idx = np.where(b_lim == params[1])[0][0]
        param_2_idx = np.where(c_lim == params[3])[0][0]
        output_grid[param_1_idx, param_2_idx] = loss

        if loss < best_loss:
            best_loss = loss
            best_params = params

        pbar.update(1)
        pbar.set_description("best loss: {} : best params : {}".format(best_loss, best_params))

    fig = plt.figure()
    plt.imshow(output_grid, cmap='hot', interpolation='nearest')

    plt.colorbar()
    plt.xlabel("c") 
    plt.ylabel("b")

    plt.xticks(np.arange(0, len(c_lim), 10), c_lim[::10])   
    plt.yticks(np.arange(0, len(b_lim), 10), b_lim[::10])
    plt.clim(0, 0.2)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Loss grid")
    plt.savefig("grid_search.png", bbox_inches='tight')


    
    print("best params", best_params)
    return best_params