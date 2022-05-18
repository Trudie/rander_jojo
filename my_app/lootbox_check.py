from scipy.stats.mstats import ttest_1samp
import numpy as np
from scipy.stats import bootstrap
from scipy.stats import chisquare

# Configs
SIGNIFICANT_LEVEL = .95
STATISTICAL_POWER = .99
SAMPLING_SIZE_UPPER_BOUND = 100000
BOOSTRAP_SAMPLE_SIZE = 100

alpha = (1-SIGNIFICANT_LEVEL)/2
beta = 1-STATISTICAL_POWER

def _bootstrap_resample(sample):
    """Bootstrap resample the sample."""
    n = len(sample)
    # bootstrap - each row is a random resample of original observations
    gen = np.random.mtrand._rand
    resample_shape = (SAMPLING_SIZE_UPPER_BOUND, BOOSTRAP_SAMPLE_SIZE)
    # random index
    i = gen.randint(0, high=n, size=resample_shape, dtype='int64')
    resamples = sample[..., i]
    return resamples

def check_one_item(drops, claim_prob):
    is_pass = True
    n_drops = len(drops)

    if n_drops >= SAMPLING_SIZE_UPPER_BOUND:
        resample_data = _bootstrap_resample(drops)
        resample_mean = np.mean(resample_data, axis=1)
        stats, pval = ttest_1samp(resample_mean, claim_prob) # by CLT, verified the mean of p_s is equal to p_a
    else:
        stats, pval = ttest_1samp(drops, claim_prob)
    if pval <= alpha:
        is_pass = False
    return is_pass

def check_pack_of_items(drops, claim_prob):
    is_pass = True
    n_drops = drops.sum()
    stats, pval = chisquare(drops, f_exp=np.multiply(claim_prob, n_drops))
    if pval <= alpha:
        is_pass = False
    return is_pass

# TODO: check pack of guarantee
def check_guarantee(drops, claim_prob, guarantee):

    for i, user in enumerate(drops):
        user_size = user.shape[0]
        if user_size >= guarantee:
            for j in range(guarantee-1, user_size, guarantee):
                if not user[j]:
                    return False

    drops_no_g = np.array([])
    for user in drops:
        user_size = user.shape[0]
        if user_size >= guarantee:
            for j in range(guarantee-1, user_size, guarantee):
                user = np.delete(user, j)
        drops_no_g = np.append(drops_no_g, user)
    return check_one_item(drops_no_g, claim_prob)

