import pytest

import numpy as np
from scipy.stats import bernoulli

from my_app.lootbox_check import  check_one_item, check_pack_of_items, check_guarantee

@pytest.mark.parametrize("n, expect", [(1000,True), (100000, False)])
def test_check_one_item(n, expect):
    claim_prob = .05
    sample_prob = .049
    # deterministic
    # n_success = int(n*sample_prob)
    # drops = np.concatenate([np.ones(n_success), np.zeros(n-n_success)])
    # synthetic data
    drops = bernoulli.rvs(sample_prob, size=n)
    assert check_one_item(drops, claim_prob) == expect


@pytest.mark.parametrize("claim_prob, sample_prob, expect", [
    ([.01,.02,.05,.1,.82],[.005,.02,.05,.1,.825], True),
    ([.01,.02,.05,.1,.82],[.01,.005,.05,.1,.835], False),
])
def test_check_pack_of_items(claim_prob, sample_prob,expect):
    # TODO: indivisible prob
    # print('verify rule: ', sum(claim_prob)==1)
    n = 1000
    # synthetic data
    drops = np.random.multinomial(n, sample_prob, size=1)[0]
    assert check_pack_of_items(drops, claim_prob) == expect

@pytest.mark.parametrize("claim_prob, sample_prob, guarantee, expect", [
    (.1,.1,15,True)
])
def test_check_guarantee(claim_prob, sample_prob, guarantee, expect):
    n=1000
    rnd_size_arr = np.random.randint(1, guarantee*3//2, size=n*2//guarantee)
    rnd_cumsum = np.cumsum(rnd_size_arr)
    # 0.1.1 adjust total size add up to n
    last_idx = (rnd_cumsum < n).sum() - 1
    rnd_size_arr = rnd_size_arr[:(last_idx+1)]
    rnd_size_arr = np.append(rnd_size_arr, n - rnd_cumsum[last_idx])
    drops = []
    for s in rnd_size_arr:
        # 0.1.2 squeeze guarantee in every idx % 15==0
        if s >= guarantee:
            user_trial = bernoulli.rvs(sample_prob, size=s-(s//guarantee))
            for j in range(guarantee-1, s, guarantee):
                user_trial = np.insert(user_trial, j, 1)
        else:
            user_trial = bernoulli.rvs(sample_prob, size=s)
        drops.append(user_trial)

    # print('verify rule: ', guarantee > round(1/claim_prob))

    assert check_guarantee(drops, claim_prob, guarantee) == expect
