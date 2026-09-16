import numpy as np
import h5py as h5

fname = "AxialICPTorch.ic.h5"

with h5.File(fname, 'r') as f:
    table = f['conserved'][:,:]


exc_sum = np.sum(table[:,7:], axis=1)
table = table[:,:8]
table[:,-1] = exc_sum

fname = "AxialICPTorch_4s.ic.h5"

with h5.File(fname, 'w') as f:
    dset = f.create_dataset("conserved", table.shape, data=table)
    f['conserved'].attrs['time'] = 0
    f['conserved'].attrs['timestep'] = 0