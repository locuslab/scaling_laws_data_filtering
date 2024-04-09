## Scaling Laws for Data Filtering

### Registering data buckets

The buckets should be registered in the following file: `all_paths_128.py`
This file contains the following information:
- `path`: The path to the data file that has the evaluation results for a model trained on that dataset.
- `samples_per_epoch_dict`: The number of samples per epoch for the corresponding dataset.
- `match_with_dict`: This tells us if the evaluation is done at a fixed epoch interval, or a fixed sample interval.
- `subsample_every_dict`: In case you want to take the average of every `k` evaluations. This is usually only useful when the evaluation is done at a fixed sample interval.

### Estimating data bucket parameters

This step involves estimating the scaling parameters for each bucket of interest. 


### Grid search to find the bucket scaling parameters

Grid search is performed to find the best scaling parameters for each bucket. The grid search is performed using the following file: `grid_search.py`. The objective minimized in the grid search is defined in `objective.py`. We chose grid search because the of instabilities observed in scipy based optimization methods.


### Objective Functions

This file implements scaling laws based on FADU, and also those inspired from work on Scaling Data Constrained Language Models.

- `func_effective_utility`: This is the function that uses the effective utility formulation as proposed in our work. 
- `func_effective_data`: This is the function that uses the formulation of effective data from Scaling Data Constrained Language Models. 

```
python process_128_grid.py --a_upper 0.02 --objective effective_utility  --d 0.1
```
Here `a_upper` is used to give an upper limit to the grid search for `a`, and `d` is the irreducibile loss. Refer to `ablations/finding_a.py` if you want to jointly minimize `a` across the pools.
Copy the obtained scaling parameters to the `results/parameter_values.py` file, and give an appropriate key name.

### Finding best bucket combination
```
python estimate_best_pool.py --key given_key_name
```