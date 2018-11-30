import numpy as np
import h5py

#from numpy import linalg as LA
from scipy import linalg as LA

# Bound State: 0,2,4
# Unbound State: 1,3,5

# Total number of bins
NBINS = 6
# List of initial bins
INIT_BINS = [4]
TARGET_BINS = [1]

# INIT_BINS = [1]
# TARGET_BINS = [4]

if INIT_BINS[0] == 4:
    CBINS = [0, 2, 4]
else:
    CBINS = [1, 3, 5]

# ABINS = [1, 3, 5, 4]
# ABINS = [0, 2, 4, 5]


# List of target bins
merged_rates = np.empty(NBINS - len(TARGET_BINS))
# Transition matrix K
K = np.array([ [0, 0, 0, 0, 0, 0 ],
      [0, 9.55068356e-01, 0, 4.49316443e-02, 0, 0],
      [0, 9.12453457e-05, 9.79353834e-01, 0, 2.05549202e-02, 0],
      [0, 1.34211520e-02, 0, 9.84648708e-01, 1.93013968e-03, 0.00000000e+00],
      [0, 0, 3.45887083e-02, 0, 9.65411292e-01, 0],
      [0, 0, 0, 0, 0, 0] ])

f = h5py.File("reweight.h5")

n_bins = 6

trans_m = np.zeros((n_bins,n_bins))

for iter in f['iterations'].keys():
    grp = f['iterations'][iter]
    rows = grp['rows']
    cols = grp['cols']
    flux = grp['flux']
    trans_m[rows, cols] += flux

# trans_m[I_ENSEMBLE,:] = 0
# trans_m[:,I_ENSEMBLE] = 0

dell = []
for row in range(n_bins):
    if trans_m[row,:].sum() == 0:
        dell.append(row)
    else:
        trans_m[row,:] /= trans_m[row,:].sum()

if False:
    print(dell)
    n_bins -= len(dell)
    # We reverse to preserve the order; 4 doesn't shift if we delete 5 first, but 5 shifts to 4 if we delete 5 first, etc.
    for row in dell[::-1]:
        trans_m = np.delete(trans_m, row, axis=0)
        trans_m = np.delete(trans_m, row, axis=1)
        for icbin, cbin in enumerate(CBINS):
            if cbin > row:
                CBINS[icbin] -= 1
            elif cbin == row:
                del CBINS[icbin]
        if INIT_BINS[0] > row:
            INIT_BINS[0] -= 1
        if TARGET_BINS[0] > row:
            TARGET_BINS[0] -= 1
    print(CBINS)
    print(trans_m)
K = np.nan_to_num(trans_m)
print(K)
# K = trans_m

# K = K[CBINS,CBINS]

eigvals, eigvecs = LA.eig(K.T)
unity = (np.abs(np.real(eigvals) - 1)).argmin()
print(eigvals, eigvecs)
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


# Hmmm.  What about...
p_dist = np.zeros(NBINS)
p_dist[INIT_BINS] = 1.0/float(len(INIT_BINS))
pp_dist = np.zeros(NBINS)
p_dist = eq_pop

p_dist = p_dist
p_dist = np.zeros((NBINS,NBINS))
p_dist = K.copy()

histogram = []

ITER = 1600000
# ITER = 0
# 100 iterations?
for i in range(ITER):
#while np.all(pp_dist == p_dist):
    #print(p_dist)
    #np_dist = K.T.dot(p_dist)
    np_dist = np.dot(K,
              p_dist - np.diag(np.diag(p_dist)))
    #histogram.append((np_dist[TARGET_BINS].sum() - p_dist[TARGET_BINS].sum())*(p_dist[INIT_BINS].sum()))
    #print(np_dist)
    histogram.append(np_dist[INIT_BINS,TARGET_BINS]*eq_pop[CBINS].sum())
    #histogram.append((np_dist[TARGET_BINS].sum())/(p_dist[I_ENSEMBLE].sum()))
    #histogram.append((np_dist[TARGET_BINS].sum() - p_dist[TARGET_BINS].sum()))
    #np_dist /= np_dist.sum()
    p_dist = np_dist
    #ITER += 1
    #print(histogram)

dt = 101

print(np.nan_to_num(histogram).shape, len(range(1,ITER+1)))
#print(ITER, (np.average(range(1,ITER+1), weights=np.nan_to_num(histogram)[:,0])*eq_pop[CBINS].sum())/11)
#print(ITER, (np.average(range(1,ITER+1), weights=np.nan_to_num(histogram)[:,0])/11))
print(ITER, (np.average(range(1,ITER+1), weights=np.nan_to_num(histogram)[:,0])/dt))
print(eq_pop)
print(eq_pop[CBINS].sum())
