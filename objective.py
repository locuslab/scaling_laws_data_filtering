import numpy as np
import copy

# d = 0.02

def power(x, power):
    if x==0:
        return 0
    return x**power


############################################################################################################
# functions for effective utility
############################################################################################################
def func_effective_utility_aggregation(samples_seen_total, params, samples_per_epoch, normalizer=100_000, d=None):
    '''
    \del y = (y * b * \delta) / n
    b_decayed = b * \delta
    \delta = base**(epochs/c)

    '''
    a_list, b_list, c_list, d = params
    # assert all a values are the same
    assert np.all(a_list == a_list[0])
    
    a = a_list[0]

    
    #get effective params for the original function
    base = 0.5
    num_epochs_full = samples_seen_total // samples_per_epoch
    # Creating arrays for each epoch and an additional one for partial epoch if exists
    epochs = np.arange(num_epochs_full + 1)
    #effective b value = b * \delta for every epoch
    all_b_decayed = []
    
    for i in range(len(a_list)):
        b, c = b_list[i], c_list[i]
        delta_list = base**(epochs/c)
        b_decayed_list = b * delta_list
        all_b_decayed.append(b_decayed_list)
    
    b_decayed_list = np.mean(all_b_decayed, axis=0)

    # Normalizing the samples
    samples = a*np.minimum(samples_per_epoch * (epochs + 1), samples_seen_total)
    samples_1 = a*samples_per_epoch * epochs
    samples, samples_1 = samples / normalizer, samples_1 / normalizer
    samples_all = copy.deepcopy(samples)

    # Calculating the loss
    loss = d + (power(samples_all[0], b_decayed_list[0]))
    if len(samples_all) > 1:
        samples, samples_1, epochs = samples[1:], samples_1[1:], epochs[1:]
        ratio = (samples/samples_1)**b_decayed_list[1:]
        loss *= ratio.prod()
    return loss


def func_effective_utility(samples_seen, params, samples_per_epoch, normalizer=100_000, d=None):
    '''
    \del y = (y * b * \delta) / n
    b_decayed = b * \delta
    \delta = base**(epochs/c)

    '''
    a, b, c, d = params
    base = 0.5
    num_epochs_full = samples_seen // samples_per_epoch
    # Creating arrays for each epoch and an additional one for partial epoch if exists
    epochs = np.arange(num_epochs_full + 1)
    

    #effective b value = b * \delta for every epoch
    delta_list = base**(epochs/c)
    b_decayed_list = b * delta_list

    # Normalizing the samples
    samples = a*np.minimum(samples_per_epoch * (epochs + 1), samples_seen)
    samples_1 = a*samples_per_epoch * epochs
    samples, samples_1 = samples / normalizer, samples_1 / normalizer
    samples_all = copy.deepcopy(samples)

    # Calculating the loss
    loss = (power(samples_all[0], b_decayed_list[0])) + d 
    if len(samples_all) > 1:
        samples, samples_1, epochs = samples[1:], samples_1[1:], epochs[1:]
        ratio = (samples/samples_1)**b_decayed_list[1:]
        loss *= ratio.prod()
    return loss

############################################################################################################
# functions for effective data with changing utility
############################################################################################################

def get_effective_samples(samples_seen, params, samples_per_epoch, normalizer=100_000):
    num_epochs_full = samples_seen // samples_per_epoch
    a, b, c, d = params
    base = 0.5
    
    # Creating arrays for each epoch and an additional one for partial epoch if exists
    epochs = np.arange(num_epochs_full + 1)
    samples = a * np.minimum(samples_per_epoch * (epochs + 1), samples_seen)
    samples_1 = a * samples_per_epoch * epochs

    # Normalizing the samples
    samples, samples_1 = samples / normalizer, samples_1 / normalizer

    # Calculating the effective samples
    effective_samples = 0
    for i in range(len(samples)):
        effective_samples += (samples[i] - samples_1[i]) * base**(epochs[i]/c)
        
    return effective_samples


def func_effective_data_aggregation(samples_seen_total, params, samples_per_epoch, normalizer=100_000):
    base = 0.5
    a_list, b_list, c_list, d = params
    samples_seen_per_bucket = samples_seen_total//len(a_list)

    # assert all a values are the same
    assert np.all(a_list == a_list[0])
    a = a_list[0]

    num_epochs_full = int(samples_seen_total // samples_per_epoch)
    epochs = np.arange(num_epochs_full + 1)
    #effective b value = b * \delta for every epoch
    all_b_decayed = []
    all_delta_list = []
    
    for i in range(len(a_list)):
        b, c = b_list[i], c_list[i]
        delta_list = base**(epochs/c)
        b_decayed_list = b * delta_list
        all_b_decayed.append(b_decayed_list)
        all_delta_list.append(delta_list)
    
    # b = (b1\delta1 + b2\delta2 + b3\delta3 + b4\delta4) / (\delta1 + \delta2 + \delta3 + \delta4)
    b_effective_list = np.sum(all_b_decayed, axis=0) / np.sum(all_delta_list, axis=0)

    # Creating arrays for each epoch and an additional one for partial epoch if exists
    samples = a*np.minimum(samples_per_epoch * (epochs + 1), samples_seen_total)
    samples_1 = a*samples_per_epoch * epochs
    samples, samples_1 = samples / normalizer, samples_1 / normalizer
    samples_all = copy.deepcopy(samples)
    
    loss = (power(samples_all[0], b_effective_list[0]))

    samples_effective_prev = 0
    for epoch in range(num_epochs_full + 1):
        # get b_effective for this epoch
        b_effective = b_effective_list[epoch]
        # get samples for this epoch by iterating over each bucket and its corresponding \delta value
        samples_effective = samples_effective_prev
        samples_in_current_epoch = samples[epoch] - samples_1[epoch]
        samples_per_bucket_in_epoch  = samples_in_current_epoch / len(a_list)
        for bucket_id in range(len(a_list)):
            delta = all_delta_list[bucket_id][epoch]
            samples_effective += samples_per_bucket_in_epoch * delta
        
        if epoch > 0:
            samples_ratio = samples_effective / samples_effective_prev
            loss *= (power(samples_ratio, b_effective))
        samples_effective_prev = samples_effective

    return loss+d

def func_effective_data(samples_seen, params, samples_per_epoch, normalizer=100_000, d=None):
    a, b, c, d = params
    effective_samples = get_effective_samples(samples_seen, params, samples_per_epoch, normalizer)
    loss =  (power(effective_samples, b)) + d
    
    return loss