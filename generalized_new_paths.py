import numpy as np
import h5py
from scipy import linalg as LA

def get_bin_lists(bins_list_name):
    while True:
        try:
            bins_list = list(int(x) for x in raw_input("Please list the %s (Ex. 0, 1, 2, 3): " % bins_list_name).split(","))
            return bins_list
        except ValueError:
            print("Please check the formatting of your input. Input should be a list of of numbers separated by commas.")


def get_nbins():
    while True:
        try:
            NBINS = int(input("Total number of bins: "))
            return NBINS
        except (NameError, TypeError):
            print("Please input a single number.")


NBINS = get_nbins()

INIT_BINS = get_bin_lists("initial bins")
if len(INIT_BINS) > NBINS:
    print("Cannot have more initial bins than bins total.")
    INIT_BINS = get_bin_lists("initial bins")

TARGET_BINS = get_bin_lists("target bins")
if len(TARGET_BINS) > NBINS:
    print("Cannot have more target bins than bins total.")
    TARGET_BINS = get_bin_lists("target bins")

"""
for bin in TARGET_BINS:
    if bin in INIT_BINS:
        print("Cannot have bins shared between target and inital states. Please reenter.")
        INIT_BINS = get_bin_lists("initial bins")
        TARGET_BINS = get_bin_lists("target bins")
        break
"""

CBINS = list(x for x in range(0, NBINS) if x not in TARGET_BINS)

# merged_rates: size = number of bins which are NOT target bins.
merged_rates = np.empty(NBINS - len(TARGET_BINS))

# Transition matrix K
# where is this got from in the general case?
K = np.array([[0, 0, 0, 0, 0, 0],
      [0, 9.55068356e-01, 0, 4.49316443e-02, 0, 0],
      [0, 9.12453457e-05, 9.79353834e-01, 0, 2.05549202e-02, 0],
      [0, 1.34211520e-02, 0, 9.84648708e-01, 1.93013968e-03, 0.00000000e+00],
      [0, 0, 3.45887083e-02, 0, 9.65411292e-01, 0],
      [0, 0, 0, 0, 0, 0]])

f = h5py.File("reweight.h5")

# empty shell for sparse trans matrix w/weights
trans_m = np.zeros((NBINS, NBINS))

for iter in f['iterations'].keys():
    grp = f['iterations'][iter]
    rows = grp['rows']
    cols = grp['cols']
    flux = grp['flux']
    trans_m[rows, cols] += flux

dell = []
for row in range(NBINS):
    if trans_m[row, :].sum() == 0:
        dell.append(row)
    else:
        trans_m[row, :] /= trans_m[row, :].sum()

# eigenvalues, eigenvectors of tranpose of K
eigvals, eigvecs = LA.eig(K.T)
unity = (np.abs(np.real(eigvals) - 1)).argmin()
print("eigenvalues", eigvals)
print("eigenvectors", eigvecs)
print("unity", unity)
eq_pop = np.abs(np.real(eigvecs)[unity])
eq_pop /= eq_pop.sum()

# probability of starting in init bin A.
distr_prob = np.random.rand(len(INIT_BINS))
paths = []
# t_bins: all bins which are not target bins.
t_bins = list(x for x in range(0, NBINS) if x not in TARGET_BINS)
# lower_bound = mfpt - error
# lower_bound = 121.8
lower_bound = 116


# What is this? This is never used
# Hmmm.  What about...
p_dist = np.zeros(NBINS)
p_dist[INIT_BINS] = 1.0/float(len(INIT_BINS))
pp_dist = np.zeros(NBINS)
p_dist = eq_pop

p_dist = p_dist
p_dist = np.zeros((NBINS, NBINS))
p_dist = K.copy()

histogram = []

ITER = 1600000
# ITER = 0
# 100 iterations?
for i in range(ITER):
    np_dist = np.dot(K,
              p_dist - np.diag(np.diag(p_dist)))
    histogram.append(np_dist[INIT_BINS, TARGET_BINS]*eq_pop[CBINS].sum())
    p_dist = np_dist

dt = 101

print(np.nan_to_num(histogram).shape, len(range(1, ITER+1)))
print(ITER, (np.average(range(1,ITER+1), weights=np.nan_to_num(histogram)[:,0])/dt))
print(eq_pop)
print(eq_pop[CBINS].sum())
