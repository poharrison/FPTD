import h5py
import numpy as np


def create_transition_matrix():
    # load HDF5 file using h5py library

    # f = h5py.File(self.reweight_file, 'r')

    f = h5py.File('reweight.h5', 'r')

    # This gives us the number of bins.  Yay!
    # this may not be true
    if len(f['bin_populations'].shape) == 3:
        # Some of these are indeed shaped differently ugh ugh ugh.
        n_bins = f['bin_populations'].shape[2]
    else:
        n_bins = f['bin_populations'].shape[1]
    trans_m = np.zeros((n_bins, n_bins))
    for iter in f['iterations'].keys():
        grp = f['iterations'][iter]
        rows = grp['rows']
        cols = grp['cols']
        flux = grp['flux']
        trans_m[rows, cols] += flux

    # normalize
    for row in range(n_bins):
        trans_m[row, :] /= trans_m[row, :].sum()
    K = np.nan_to_num(trans_m)
    return K


create_transition_matrix()