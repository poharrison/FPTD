# import multiprocessing
import numpy as np
from multiprocessing import Pool

# Total number of bins
NBINS = 3
# List of initial bins
INIT_BINS = [0]
# List of target bins
TARGET_BINS = [2]
merged_rates = np.empty(NBINS - len(TARGET_BINS))
# Transition matrix K
K = np.array([[0.43871389, 0.29933964, 0.87257363], [0.78913289, 0.23298078, 0.99836441],
                 [0.36616367, 0.13101896, 0.57178222]])

# probability of starting in init bin A.
distr_prob = np.random.rand(len(INIT_BINS))
paths = []
# t_bins: all bins which are not target bins.
t_bins = list(x for x in range(0, NBINS) if x not in TARGET_BINS)
# lower_bound = mfpt - error
lower_bound = .9

# Merge bins in target state.
def merge_rates():
    for i in range(0, NBINS):
        if i not in TARGET_BINS:
            for j in TARGET_BINS:
                merged_rates[i] += K[i, j]


# loc: bin at which the path end is currently located
# rate: probability of getting to bin loc by some path
# returns rate * probabilty of going from bin loc to target state
def end(rate, loc):
    return rate * merged_rates[loc]

# loc: bin at which the path end is currently located
# rate: probability of getting to bin loc by some path
# returns a list of tuples: (probability of going from bin loc to bin b (along current path), bin b)
# ^ for all bins b not in target state
def walk(rate, loc):
    return list((rate * K[loc, b], b) for b in t_bins)

# Calculate the effective MFPT (MFPT of sampled paths).
def calc_emfpt(paths=None, emfpt=0):
    path_sum = np.sum(paths)
    return emfpt + path_sum * len(paths)


if __name__ == '__main__':
    merge_rates()
    # For N = 1, probability is simply the rate of transitioning from the initial bin
    # to any bin in target state * the probability of starting in the initial bin
    # rates: a list of probabilities and bins (probability of getting to bin by some path, # of that bin)
    rates = list((distr_prob[i], i) for i in INIT_BINS)
    i = 0
    emfpt = 0
    # If emfpt < lower_bound, emfpt is in range of actual MFPT (Implies that enough paths have been sampled s.t.
    # distribution can be considered accurate and complete.)
    while emfpt < lower_bound:
        with Pool() as p:  # arg: processes=5
            # Calculate probability of current paths going from current bin to target state.
            paths.append(p.starmap(end, rates))
            # Calculate probability of current paths going from current bin to other bins not in target state.
            # (ie. Calculate probability of path length increasing by 1)
            rates = p.starmap(walk, rates)
            temp = []
            for rate in rates:
                for item in rate:
                    temp.append(item)
            rates = temp
            # Update emfpt to include newly sampled paths.
            emfpt = calc_emfpt(paths[i], emfpt)
            i += 1
            print(rates)
            print(paths)
    print(paths)
