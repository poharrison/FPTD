# import multiprocessing
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp

NBINS = 3
INIT_BINS = [0]
TARGET_BINS = [2]
merged_rates = np.empty(NBINS - len(TARGET_BINS))
K = np.array([[0.43871389, 0.29933964, 0.87257363], [0.78913289, 0.23298078, 0.99836441],
                 [0.36616367, 0.13101896, 0.57178222]])

# probability of starting in init bin A.
distr_prob = np.random.rand(len(INIT_BINS))
paths = []
# t_bins: all bins which are not target bins
t_bins = list(x for x in range(0, NBINS) if x not in TARGET_BINS)
# lower_bound = mfpt - error
lower_bound = .8


def merge_rates():
    for i in range(0, NBINS):
        if i not in TARGET_BINS:
            for j in TARGET_BINS:
                merged_rates[i] += K[i, j]


def end(rate, loc):
    return rate * merged_rates[loc]


def walk(rate, loc):
    return list((rate * K[loc, b], b) for b in t_bins)


def calc_mfpt(paths=None, emfpt=0):
    path_sum = np.sum(paths)
    return emfpt + path_sum * len(paths)


if __name__ == '__main__':
    merge_rates()
    # for N = 1, probability is simply the rate of transitioning from the initial bin
    # to any bin in target state * the probability of starting in the initial bin
    rates = list((distr_prob[i], i) for i in INIT_BINS)
    i = 0
    emfpt = 0
    while emfpt < lower_bound:
        with Pool() as p:  # arg: processes=5
            paths.append(p.starmap(end, rates))
            rates = p.starmap(walk, rates)
            temp = []
            for rate in rates:
                for item in rate:
                    temp.append(item)
            rates = temp
            emfpt = calc_mfpt(paths[i], emfpt)
            i += 1
    print(paths)
