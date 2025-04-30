import numpy as np
import h5py as h5
import os, sys

template_mesh_file = "trevilo-cases/torch_tst_2023/Torch.mesh.h5"
finer_fname = "trevilo-cases/torch_tst_2023/Torch_Fine.mesh.h5"
coarser_fname = "trevilo-cases/torch_tst_2023/Torch_Coarse.mesh.h5"

# Additionally, Adjust the initial condition files
template_ic_file = "/usr/workspace/bedonian1/mean_r6/output/AxialICPTorch-00060000.h5"
finer_ic_fname = "/usr/workspace/bedonian1/mean_r6_fine/output/AxialICPTorch-00060000.h5"
coarser_ic_fname = "/usr/workspace/bedonian1/mean_r6_coarse/output/AxialICPTorch-00060000.h5"

with h5.File(template_mesh_file, 'r') as f:
    # table = f['conserved'][:,:]

    temp_rad = f['radius'][:]
    temp_x = f['x'][:]


with h5.File(template_ic_file, 'r') as f:
    # table = f['conserved'][:,:]

    temp_cons = f['conserved'][:,:]
    temp_time = f['conserved'].attrs['time']
    temp_ts = f['conserved'].attrs['timestep']



temp_N = len(temp_x)
# Coarse: Remove every other point on the interior
c_mask = np.arange(2, temp_N, 2)
c_rad = np.delete(temp_rad, c_mask)
c_x = np.delete(temp_x, c_mask)
c_cons = np.delete(temp_cons, c_mask, axis=0)

# Fine: Linear interpolation between interior points
f_x = np.zeros(2*temp_N-1)
f_rad = np.zeros(2*temp_N-1)
f_cons = np.zeros([2*temp_N-1, 11]) # hard code number of vars
for i in range(temp_N-1):
    f_x[2*i] = temp_x[i]
    f_rad[2*i] = temp_rad[i]
    f_cons[2*i, :] = temp_cons[i, :]
    f_x[2*i+1] = (temp_x[i] + temp_x[i+1])/2
    f_rad[2*i+1] = (temp_rad[i] + temp_rad[i+1])/2
    f_cons[2*i+1, :] = (temp_cons[i, :] + temp_cons[i+1, :])/2
f_x[-1] = temp_x[-1]
f_rad[-1] = temp_rad[-1]
f_cons[-1, :] = temp_cons[-1, :]


with h5.File(coarser_fname, 'w') as f:
    dset = f.create_dataset("radius", c_rad.shape, data=c_rad)
    dset = f.create_dataset("x", c_x.shape, data=c_x)

with h5.File(finer_fname, 'w') as f:
    dset = f.create_dataset("radius", f_rad.shape, data=f_rad)
    dset = f.create_dataset("x", f_x.shape, data=f_x)

with h5.File(coarser_ic_fname, 'w') as f:
    dset = f.create_dataset("conserved", c_cons.shape, data=c_cons)
    f['conserved'].attrs['time'] = temp_time
    f['conserved'].attrs['timestep'] = temp_ts

with h5.File(finer_ic_fname, 'w') as f:
    dset = f.create_dataset("conserved", f_cons.shape, data=f_cons)
    f['conserved'].attrs['time'] = temp_time
    f['conserved'].attrs['timestep'] = temp_ts
