import h5py as h5
import numpy as np


source_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_cold/cold_unrun_p3_restart_output-torch.sol.h5"
result_restart = "/g/g14/bedonian1/bedonian1/mean_tps2d_cold/nocold_unrun_p3_restart_output-torch.sol.h5"

uy_uniform = 0.0
T_uniform = 300.0

s_attr = {}
with h5.File(source_restart, "r") as f:
    for key in f.attrs.keys():
        s_attr[key] = f.attrs[key]

    s_swirl = np.copy(f['swirl']['swirl'])
    s_temperature = np.copy(f['temperature']['temperature'])
    s_velx = np.copy(f['velocity']['x-comp'])
    s_vely = np.copy(f['velocity']['y-comp'])

s_attr['time'] = 0.0
s_attr['iteration'] = 0
s_swirl[:] = 0.0
s_temperature[:] = T_uniform
s_velx[:] = 0.0
s_vely[:] = uy_uniform

with h5.File(result_restart, "w") as f:
    for key in s_attr.keys():
        f.attrs[key] = s_attr[key]

    g1 = f.create_group("swirl")
    g1.create_dataset("swirl", s_swirl.shape, data=s_swirl)
    g2 = f.create_group("temperature")
    g2.create_dataset("temperature", s_temperature.shape, data=s_temperature)
    g3 = f.create_group("velocity")
    g3.create_dataset("x-comp", s_velx.shape, data=s_velx)
    g3.create_dataset("y-comp", s_vely.shape, data=s_vely)